from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
def _cx_cost(clifford):
    """Return the number of CX gates required for Clifford decomposition."""
    if clifford.num_qubits == 2:
        return _cx_cost2(clifford)
    if clifford.num_qubits == 3:
        return _cx_cost3(clifford)
    raise Exception('No Clifford CX cost function for num_qubits > 3.')