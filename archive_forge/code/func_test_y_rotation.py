import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('gate,trans_y', ((cirq.SingleQubitCliffordGate.I, (cirq.Y, False)), (cirq.SingleQubitCliffordGate.H, (cirq.Y, True)), (cirq.SingleQubitCliffordGate.X, (cirq.Y, True)), (cirq.SingleQubitCliffordGate.Y, (cirq.Y, False)), (cirq.SingleQubitCliffordGate.Z, (cirq.Y, True)), (cirq.SingleQubitCliffordGate.X_sqrt, (cirq.Z, False)), (cirq.SingleQubitCliffordGate.X_nsqrt, (cirq.Z, True)), (cirq.SingleQubitCliffordGate.Y_sqrt, (cirq.Y, False)), (cirq.SingleQubitCliffordGate.Y_nsqrt, (cirq.Y, False)), (cirq.SingleQubitCliffordGate.Z_sqrt, (cirq.X, True)), (cirq.SingleQubitCliffordGate.Z_nsqrt, (cirq.X, False))))
def test_y_rotation(gate, trans_y):
    assert gate.pauli_tuple(cirq.Y) == trans_y