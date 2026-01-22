from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _to_chi(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Chi representation."""
    if rep == 'Chi':
        return data
    _check_nqubit_dim(input_dim, output_dim)
    if rep == 'Operator':
        return _from_operator('Chi', data, input_dim, output_dim)
    if rep != 'Choi':
        data = _to_choi(rep, data, input_dim, output_dim)
    return _choi_to_chi(data, input_dim)