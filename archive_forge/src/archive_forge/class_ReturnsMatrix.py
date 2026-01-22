from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
class ReturnsMatrix(cirq.Gate):

    def _unitary_(self) -> np.ndarray:
        return m1

    def num_qubits(self):
        return 1