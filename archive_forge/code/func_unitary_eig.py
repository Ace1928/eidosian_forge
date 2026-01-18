import cmath
import math
from typing import (
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq._import import LazyLoader
from cirq.linalg import combinators, diagonalize, predicates, transformations
def unitary_eig(matrix: np.ndarray, check_preconditions: bool=True, atol: float=1e-08) -> Tuple[np.ndarray, np.ndarray]:
    """Gives the guaranteed unitary eigendecomposition of a normal matrix.

    All hermitian and unitary matrices are normal matrices. This method was
    introduced as for certain classes of unitary matrices (where the eigenvalues
    are close to each other) the eigenvectors returned by `numpy.linalg.eig` are
    not guaranteed to be orthogonal.
    For more information, see https://github.com/numpy/numpy/issues/15461.

    Args:
        matrix: A normal matrix. If not normal, this method is not
            guaranteed to return correct eigenvalues.  A normal matrix
            is one where $A A^\\dagger = A^\\dagger A$.
        check_preconditions: When true and matrix is not unitary,
            a `ValueError` is raised when the matrix is not normal.
        atol: The absolute tolerance when checking whether the original matrix
            was unitary.

    Returns:
        A Tuple of
            eigvals: The eigenvalues of `matrix`.
            V: The unitary matrix with the eigenvectors as columns.

    Raises:
        ValueError: if the input matrix is not normal.
    """
    if check_preconditions and (not predicates.is_normal(matrix, atol=atol)):
        raise ValueError(f'Input must correspond to a normal matrix .Received input:\n{matrix}')
    R, V = linalg.schur(matrix, output='complex')
    return (R.diagonal(), V)