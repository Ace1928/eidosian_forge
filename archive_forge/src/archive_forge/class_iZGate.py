import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
class iZGate(cirq.testing.SingleQubitGate):
    """Equivalent to an iZ gate without _act_on_ defined on it."""

    def _unitary_(self):
        return np.array([[1j, 0], [0, -1j]])