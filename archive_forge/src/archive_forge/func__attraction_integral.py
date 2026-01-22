import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def _attraction_integral(*args):
    """Compute the electron-nuclear attraction integral for two contracted Gaussian functions.

        Args:
            *args (array[float]): initial values of the differentiable parameters

        Returns:
            array[float]: the electron-nuclear attraction integral
        """
    if r.requires_grad:
        coor = args[0]
        args_a = [arg[0] for arg in args[1:]]
        args_b = [arg[1] for arg in args[1:]]
    else:
        coor = r
        args_a = [arg[0] for arg in args]
        args_b = [arg[1] for arg in args]
    alpha, ca, ra = _generate_params(basis_a.params, args_a)
    beta, cb, rb = _generate_params(basis_b.params, args_b)
    if basis_a.params[1].requires_grad or normalize:
        ca = ca * primitive_norm(basis_a.l, alpha)
        cb = cb * primitive_norm(basis_b.l, beta)
        na = contracted_norm(basis_a.l, alpha, ca)
        nb = contracted_norm(basis_b.l, beta, cb)
    else:
        na = nb = 1.0
    v = na * nb * (ca * cb[:, np.newaxis] * nuclear_attraction(basis_a.l, basis_b.l, ra, rb, alpha, beta[:, np.newaxis], coor)).sum()
    return v