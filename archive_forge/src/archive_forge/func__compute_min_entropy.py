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
def _compute_min_entropy(density_matrix, base):
    """Compute the minimum entropy from a density matrix

    Args:
        density_matrix (tensor_like): ``(2**N, 2**N)`` tensor density matrix for an integer `N`.
        base (float, int): Base for the logarithm. If None, the natural logarithm is used.

    Returns:
        float: Minimum entropy of the density matrix.

    **Example**

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_min_entropy(x)
    0.6931472

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_min_entropy(x, base=2)
    1.0
    """
    div_base = np.log(base) if base else 1
    evs, _ = qml.math.linalg.eigh(density_matrix)
    evs = qml.math.real(evs)
    minimum_entropy = -qml.math.log(qml.math.max(evs)) / div_base
    return minimum_entropy