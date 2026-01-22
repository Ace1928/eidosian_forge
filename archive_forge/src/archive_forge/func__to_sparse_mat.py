import warnings
from copy import copy
from functools import reduce, lru_cache
from typing import Iterable
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane import math
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd, Sum
def _to_sparse_mat(self, wire_order, buffer_size=None):
    """Compute the sparse matrix of the Pauli sentence by efficiently adding the Pauli words
        that it is composed of. See pauli_sparse_matrices.md for the technical details."""
    pauli_words = list(self)
    n_wires = len(wire_order)
    matrix_size = 2 ** n_wires
    matrix = sparse.csr_matrix((matrix_size, matrix_size), dtype='complex128')
    op_sparse_idx = _ps_to_sparse_index(pauli_words, wire_order)
    _, unique_sparse_structures, unique_invs = np.unique(op_sparse_idx, axis=0, return_index=True, return_inverse=True)
    pw_sparse_structures = unique_sparse_structures[unique_invs]
    buffer_size = buffer_size or 2 ** 30
    buffer_size = max(1, buffer_size // ((16 + 8) * matrix_size))
    mat_data = np.empty((matrix_size, buffer_size), dtype=np.complex128)
    mat_indices = np.empty((matrix_size, buffer_size), dtype=np.int64)
    n_matrices_in_buffer = 0
    for sparse_structure in unique_sparse_structures:
        indices, *_ = np.nonzero(pw_sparse_structures == sparse_structure)
        mat = self._sum_same_structure_pws([pauli_words[i] for i in indices], wire_order)
        mat_data[:, n_matrices_in_buffer] = mat.data
        mat_indices[:, n_matrices_in_buffer] = mat.indices
        n_matrices_in_buffer += 1
        if n_matrices_in_buffer == buffer_size:
            matrix += self._sum_different_structure_pws(mat_indices, mat_data)
            n_matrices_in_buffer = 0
    matrix += self._sum_different_structure_pws(mat_indices[:, :n_matrices_in_buffer], mat_data[:, :n_matrices_in_buffer])
    matrix.eliminate_zeros()
    return matrix