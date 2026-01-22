from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
def _cx_cost2(clifford):
    """Return CX cost of a 2-qubit clifford."""
    U = clifford.tableau[:, :-1]
    r00 = _rank2(U[0, 0], U[0, 2], U[2, 0], U[2, 2])
    r01 = _rank2(U[0, 1], U[0, 3], U[2, 1], U[2, 3])
    if r00 == 2:
        return r01
    return r01 + 1 - r00