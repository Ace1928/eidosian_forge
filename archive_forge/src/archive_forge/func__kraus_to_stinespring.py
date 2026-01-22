from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _kraus_to_stinespring(data, input_dim, output_dim):
    """Transform Kraus representation to Stinespring representation."""
    stine_pair = [None, None]
    for i, kraus in enumerate(data):
        if kraus is not None:
            num_kraus = len(kraus)
            stine = np.zeros((output_dim * num_kraus, input_dim), dtype=complex)
            for j, mat in enumerate(kraus):
                vec = np.zeros(num_kraus)
                vec[j] = 1
                stine += np.kron(mat, vec[:, None])
            stine_pair[i] = stine
    return tuple(stine_pair)