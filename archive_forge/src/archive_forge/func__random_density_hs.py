from __future__ import annotations
from typing import Literal
import numpy as np
from numpy.random import default_rng
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.random import random_unitary
from .statevector import Statevector
from .densitymatrix import DensityMatrix
def _random_density_hs(dim, rank, seed):
    """
    Generate a random density matrix from the Hilbert-Schmidt metric.

    Args:
        dim (int): the dimensions of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int or np.random.Generator): default rng.

    Returns:
        ndarray: rho (N,N)  a density matrix.
    """
    mat = _ginibre_matrix(dim, rank, seed)
    mat = mat.dot(mat.conj().T)
    return mat / np.trace(mat)