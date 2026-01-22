import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def _overlap_integral(*args):
    """Normalize and compute the overlap integral for two contracted Gaussian functions.

        Args:
            *args (array[float]): initial values of the differentiable parameters

        Returns:
            array[float]: the overlap integral between two contracted Gaussian orbitals
        """
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
    return na * nb * (ca[:, np.newaxis] * cb * gaussian_overlap(basis_a.l, basis_b.l, ra, rb, alpha[:, np.newaxis], beta)).sum()