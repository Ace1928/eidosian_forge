import pytest
import numpy as np
import cirq
class NotPhaser:

    def _unitary_(self):
        return np.array([[0, 1], [1, 0]])

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        return NotImplemented