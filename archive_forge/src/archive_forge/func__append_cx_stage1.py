import numpy as np
from qiskit.circuit import QuantumCircuit
def _append_cx_stage1(qc, n):
    """A single layer of CX gates."""
    for i in range(n // 2):
        qc.cx(2 * i, 2 * i + 1)
    for i in range((n + 1) // 2 - 1):
        qc.cx(2 * i + 2, 2 * i + 1)
    return qc