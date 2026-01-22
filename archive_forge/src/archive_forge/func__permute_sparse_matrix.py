import itertools
from functools import reduce
from typing import Generator, Iterable, Tuple
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
import pennylane as qml
from pennylane.wires import Wires
def _permute_sparse_matrix(matrix, wires, wire_order):
    """Permute the matrix to match the wires given in `wire_order`.

    Args:
        matrix (scipy.sparse.spmatrix): matrix to permute
        wires (list): wires determining the subspace that base matrix acts on; a base matrix of
            dimension :math:`2^n` acts on a subspace of :math:`n` wires
        wire_order (list): global wire order, which has to contain all wire labels in ``wires``,
            but can also contain additional labels

    Returns:
        scipy.sparse.spmatrix: permuted matrix
    """
    U = _permutation_sparse_matrix(wires, wire_order)
    if U is not None:
        matrix = U.T @ matrix @ U
        matrix.eliminate_zeros()
    return matrix