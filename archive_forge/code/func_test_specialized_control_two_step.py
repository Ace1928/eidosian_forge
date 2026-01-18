import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
@pytest.mark.parametrize('input_gate, specialized_output', [(cirq.Z, cirq.CCZ), (cirq.X, cirq.CCX), (cirq.ZPowGate(exponent=0.5), cirq.CCZPowGate(exponent=0.5)), (cirq.XPowGate(exponent=0.5), cirq.CCXPowGate(exponent=0.5))])
def test_specialized_control_two_step(input_gate, specialized_output):
    assert input_gate.controlled().controlled() == specialized_output
    assert input_gate.controlled(num_controls=2) == specialized_output
    assert input_gate.controlled(control_values=[1, 1]) == specialized_output
    assert input_gate.controlled(control_values=cirq.SumOfProducts([[1, 1]])) == specialized_output
    assert input_gate.controlled(control_qid_shape=(2, 2)) == specialized_output
    assert np.allclose(cirq.unitary(specialized_output), cirq.unitary(cirq.ControlledGate(input_gate, num_controls=2)))