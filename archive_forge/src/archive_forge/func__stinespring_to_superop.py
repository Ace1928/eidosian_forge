from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _stinespring_to_superop(data, input_dim, output_dim):
    """Transform Stinespring representation to SuperOp representation."""
    trace_dim = data[0].shape[0] // output_dim
    stine_l = np.reshape(data[0], (output_dim, trace_dim, input_dim))
    if data[1] is None:
        stine_r = stine_l
    else:
        stine_r = np.reshape(data[1], (output_dim, trace_dim, input_dim))
    return np.reshape(np.einsum('iAj,kAl->ikjl', stine_r.conj(), stine_l), (output_dim * output_dim, input_dim * input_dim))