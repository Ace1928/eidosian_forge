import numpy as np
import pytest
import cirq
class UnitaryHGate(cirq.Gate):

    def num_qubits(self) -> int:
        return 1

    def _unitary_(self):
        return np.array([[1, 1], [1, -1]]) / 2 ** 0.5