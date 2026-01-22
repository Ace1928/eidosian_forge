from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _ptm_to_superop(data, input_dim):
    """Transform PTM representation to SuperOp representation."""
    num_qubits = int(np.log2(input_dim))
    return _transform_from_pauli(data, num_qubits)