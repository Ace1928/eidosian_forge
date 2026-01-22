from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _stinespring_to_kraus(data, output_dim):
    """Transform Stinespring representation to Kraus representation."""
    kraus_pair = []
    for stine in data:
        if stine is None:
            kraus_pair.append(None)
        else:
            trace_dim = stine.shape[0] // output_dim
            iden = np.eye(output_dim)
            kraus = []
            for j in range(trace_dim):
                vec = np.zeros(trace_dim)
                vec[j] = 1
                kraus.append(np.kron(iden, vec[None, :]).dot(stine))
            kraus_pair.append(kraus)
    return tuple(kraus_pair)