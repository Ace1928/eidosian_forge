import functools
import itertools
from string import ascii_letters as ABC
from autoray import numpy as np
from numpy import float64
import pennylane as qml
from . import single_dispatch  # pylint:disable=unused-import
from .matrix_manipulation import _permute_dense_matrix
from .multi_dispatch import diag, dot, scatter_element_add, einsum, get_interface
from .utils import is_abstract, allclose, cast, convert_like, cast_like
def _check_density_matrix(density_matrix):
    """Check the shape, the trace and the positive semi-definitiveness of a matrix."""
    dim = density_matrix.shape[-1]
    if len(density_matrix.shape) not in (2, 3) or density_matrix.shape[-2] != dim or (not np.log2(dim).is_integer()):
        raise ValueError('Density matrix must be of shape (2**N, 2**N) or (batch_dim, 2**N, 2**N).')
    if len(density_matrix.shape) == 2:
        density_matrix = qml.math.stack([density_matrix])
    if not is_abstract(density_matrix):
        for dm in density_matrix:
            trace = np.trace(dm)
            if not allclose(trace, 1.0, atol=1e-10):
                raise ValueError('The trace of the density matrix should be one.')
            conj_trans = np.transpose(np.conj(dm))
            if not allclose(dm, conj_trans):
                raise ValueError('The matrix is not Hermitian.')
            evs, _ = qml.math.linalg.eigh(dm)
            evs = np.real(evs)
            evs_non_negative = [ev for ev in evs if ev >= -1e-07]
            if len(evs) != len(evs_non_negative):
                raise ValueError('The matrix is not positive semi-definite.')