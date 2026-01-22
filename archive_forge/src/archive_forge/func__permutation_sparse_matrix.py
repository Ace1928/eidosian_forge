import itertools
from functools import reduce
from typing import Generator, Iterable, Tuple
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
import pennylane as qml
from pennylane.wires import Wires
def _permutation_sparse_matrix(expanded_wires: Iterable, wire_order: Iterable) -> csr_matrix:
    """Helper function which generates a permutation matrix in sparse format that swaps the wires
    in ``expanded_wires`` to match the order given by the ``wire_order`` argument.

    Args:
        expanded_wires (Iterable): inital wires
        wire_order (Iterable): final wires

    Returns:
        csr_matrix: permutation matrix in CSR sparse format
    """
    n_total_wires = len(wire_order)
    U = None
    for i in range(n_total_wires):
        if expanded_wires[i] != wire_order[i]:
            if U is None:
                U = eye(2 ** n_total_wires, format='csr')
            j = expanded_wires.index(wire_order[i])
            U = U @ _sparse_swap_mat(i, j, n_total_wires)
            U.eliminate_zeros()
            expanded_wires[i], expanded_wires[j] = (expanded_wires[j], expanded_wires[i])
    return U