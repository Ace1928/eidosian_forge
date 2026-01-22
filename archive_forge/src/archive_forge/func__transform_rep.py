from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _transform_rep(input_rep, output_rep, data, input_dim, output_dim):
    """Transform a QuantumChannel between representation."""
    if input_rep == output_rep:
        return data
    if output_rep == 'Choi':
        return _to_choi(input_rep, data, input_dim, output_dim)
    if output_rep == 'Operator':
        return _to_operator(input_rep, data, input_dim, output_dim)
    if output_rep == 'SuperOp':
        return _to_superop(input_rep, data, input_dim, output_dim)
    if output_rep == 'Kraus':
        return _to_kraus(input_rep, data, input_dim, output_dim)
    if output_rep == 'Chi':
        return _to_chi(input_rep, data, input_dim, output_dim)
    if output_rep == 'PTM':
        return _to_ptm(input_rep, data, input_dim, output_dim)
    if output_rep == 'Stinespring':
        return _to_stinespring(input_rep, data, input_dim, output_dim)
    raise QiskitError(f'Invalid QuantumChannel {output_rep}')