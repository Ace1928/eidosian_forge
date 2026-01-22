import pytest
import numpy as np
import sympy
import cirq
class BadGateDecompose(cirq.testing.SingleQubitGate):

    def _decompose_(self, qubits):
        return cirq.Y(qubits[0])

    def _unitary_(self):
        return np.array([[0, 1], [1, 0]])