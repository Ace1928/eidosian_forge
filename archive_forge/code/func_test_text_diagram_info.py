import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('gate,sym,exp', ((cirq.SingleQubitCliffordGate.I, 'I', 1), (cirq.SingleQubitCliffordGate.H, 'H', 1), (cirq.SingleQubitCliffordGate.X, 'X', 1), (cirq.SingleQubitCliffordGate.X_sqrt, 'X', 0.5), (cirq.SingleQubitCliffordGate.X_nsqrt, 'X', -0.5), (cirq.SingleQubitCliffordGate.from_xz_map((cirq.Y, False), (cirq.X, True)), '(X^-0.5-Z^0.5)', 1)))
def test_text_diagram_info(gate, sym, exp):
    assert cirq.circuit_diagram_info(gate) == cirq.CircuitDiagramInfo(wire_symbols=(sym,), exponent=exp)