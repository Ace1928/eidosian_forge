from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _check_nqubit_dim(input_dim, output_dim):
    """Return true if dims correspond to an n-qubit channel."""
    if input_dim != output_dim:
        raise QiskitError(f'Not an n-qubit channel: input_dim ({input_dim}) != output_dim ({output_dim})')
    num_qubits = int(np.log2(input_dim))
    if 2 ** num_qubits != input_dim:
        raise QiskitError('Not an n-qubit channel: input_dim != 2 ** n')