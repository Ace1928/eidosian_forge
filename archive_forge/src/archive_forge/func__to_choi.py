from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _to_choi(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Choi representation."""
    if rep == 'Choi':
        return data
    if rep == 'Operator':
        return _from_operator('Choi', data, input_dim, output_dim)
    if rep == 'SuperOp':
        return _superop_to_choi(data, input_dim, output_dim)
    if rep == 'Kraus':
        return _kraus_to_choi(data)
    if rep == 'Chi':
        return _chi_to_choi(data, input_dim)
    if rep == 'PTM':
        data = _ptm_to_superop(data, input_dim)
        return _superop_to_choi(data, input_dim, output_dim)
    if rep == 'Stinespring':
        return _stinespring_to_choi(data, input_dim, output_dim)
    raise QiskitError(f'Invalid QuantumChannel {rep}')