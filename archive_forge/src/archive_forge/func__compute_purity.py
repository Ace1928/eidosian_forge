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
def _compute_purity(density_matrix):
    """Compute the purity from a density matrix

    Args:
        density_matrix (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` tensor for an integer `N`.

    Returns:
        float: Purity of the density matrix.

    **Example**

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_purity(x)
    0.5

    >>> x = [[1/2, 1/2], [1/2, 1/2]]
    >>> _compute_purity(x)
    1

    """
    batched = len(qml.math.shape(density_matrix)) > 2
    if batched:
        return qml.math.real(qml.math.einsum('abc,acb->a', density_matrix, density_matrix))
    return qml.math.real(qml.math.einsum('ab,ba', density_matrix, density_matrix))