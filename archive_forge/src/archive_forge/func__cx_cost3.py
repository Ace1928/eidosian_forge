from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
def _cx_cost3(clifford):
    """Return CX cost of a 3-qubit clifford."""
    U = clifford.tableau[:, :-1]
    n = 3
    R1 = np.zeros((n, n), dtype=int)
    R2 = np.zeros((n, n), dtype=int)
    for q1 in range(n):
        for q2 in range(n):
            R2[q1, q2] = _rank2(U[q1, q2], U[q1, q2 + n], U[q1 + n, q2], U[q1 + n, q2 + n])
            mask = np.zeros(2 * n, dtype=int)
            mask[[q2, q2 + n]] = 1
            isLocX = np.array_equal(U[q1, :] & mask, U[q1, :])
            isLocZ = np.array_equal(U[q1 + n, :] & mask, U[q1 + n, :])
            isLocY = np.array_equal((U[q1, :] ^ U[q1 + n, :]) & mask, U[q1, :] ^ U[q1 + n, :])
            R1[q1, q2] = 1 * (isLocX or isLocZ or isLocY) + 1 * (isLocX and isLocZ and isLocY)
    diag1 = np.sort(np.diag(R1)).tolist()
    diag2 = np.sort(np.diag(R2)).tolist()
    nz1 = np.count_nonzero(R1)
    nz2 = np.count_nonzero(R2)
    if diag1 == [2, 2, 2]:
        return 0
    if diag1 == [1, 1, 2]:
        return 1
    if diag1 == [0, 1, 1] or (diag1 == [1, 1, 1] and nz2 < 9) or (diag1 == [0, 0, 2] and diag2 == [1, 1, 2]):
        return 2
    if diag1 == [1, 1, 1] and nz2 == 9 or (diag1 == [0, 0, 1] and (nz1 == 1 or diag2 == [2, 2, 2] or (diag2 == [1, 1, 2] and nz2 < 9))) or (diag1 == [0, 0, 2] and diag2 == [0, 0, 2]) or (diag2 == [1, 2, 2] and nz1 == 0):
        return 3
    if diag2 == [0, 0, 1] or (diag1 == [0, 0, 0] and (diag2 == [1, 1, 1] and nz2 == 9 and (nz1 == 3) or (diag2 == [0, 1, 1] and nz2 == 8 and (nz1 == 2)))):
        return 5
    if nz1 == 3 and nz2 == 3:
        return 6
    return 4