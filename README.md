# Physics-Informed Neural Networks (PINNs) for the 1D Burgers' Equation

This repository contains a PyTorch implementation of Physics-Informed Neural Networks (PINNs) applied to the 1D viscous Burgers' equation, following the seminal work by **Raissi, Perdikaris, and Karniadakis (2019)**.

## 📌 Problem Overview
The 1D Burgers' equation is a fundamental partial differential equation (PDE) from fluid mechanics that combines nonlinear propagation and diffusion effects:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \frac{0.01}{\pi} \frac{\partial^2 u}{\partial x^2} = 0, \quad x \in [-1, 1], \ t \in [0, 1]$$

 subject to initial and boundary conditions.

## 🚀 Key Features & Methodology
- **Architecture:** Feedforward Neural Network (Multi-Layer Perceptron) with Tanh activation functions.
- **Collocation Points:** Generated using **Latin Hypercube Sampling (LHS)** for robust domain exploration.
- **Optimization Strategy:** Two-phase training combining **Adam optimizer** (for initial convergence) followed by **L-BFGS optimizer** (for high-precision fine-tuning).
- **Automatic Differentiation:** Utilizes PyTorch's `torch.autograd` to compute exact PDE residuals without numerical discretization errors.

## 📊 Performance & Benchmarking
The model is benchmarked against the official analytical solution (`burgers_shock.mat` from the original PINNs repository):

| Baseline Method | Relative $L_2$ Error | Description |
| :--- | :--- | :--- |
| **Numerical Baseline** (Custom FD) | $\approx 6.89 \times 10^{-2}$ | Introduces minor numerical diffusion around shock fronts. |
| **Analytical Baseline** (`burgers_shock.mat`) | $\approx 1.11 \times 10^{-3}$ | High-precision comparison matching publication standards. |

## 📈 Visualizations
The model successfully captures the shockwave dynamics and reproduces the exact temporal slices ($t = 0.25, 0.50, 0.75$) matching Figure A.6 of the reference paper.

## 🛠️ Requirements
- Python 3.x
- PyTorch
- NumPy
- SciPy
- Matplotlib

## 📜 References
- M. Raissi, P. Perdikaris, G. E. Karniadakis, *"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"* (2019).
