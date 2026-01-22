import numpy as np
import pytest
import cirq
class GoodGateExplicitPauliExpansion(cirq.testing.SingleQubitGate):

    def _unitary_(self) -> np.ndarray:
        return np.sqrt(1 / 2) * X + np.sqrt(1 / 3) * Y + np.sqrt(1 / 6) * Z

    def _pauli_expansion_(self) -> cirq.LinearDict[str]:
        return cirq.LinearDict({'X': np.sqrt(1 / 2), 'Y': np.sqrt(1 / 3), 'Z': np.sqrt(1 / 6)})