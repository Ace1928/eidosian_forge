import numpy as np
from qiskit.circuit import QuantumCircuit
def _odd_pattern1(n):
    """A pattern denoted by Pj in [1] for odd number of qubits:
    [n-2, n-4, n-4, ..., 3, 3, 1, 1, 0, 0, 2, 2, ..., n-3, n-3]
    """
    pat = []
    pat.append(n - 2)
    for i in range((n - 3) // 2):
        pat.append(n - 2 * i - 4)
        pat.append(n - 2 * i - 4)
    for i in range((n - 1) // 2):
        pat.append(2 * i)
        pat.append(2 * i)
    return pat