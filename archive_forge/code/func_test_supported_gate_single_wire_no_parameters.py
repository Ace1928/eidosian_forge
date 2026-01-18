from math import sqrt, pi
import pytest
import numpy as np
from flaky import flaky
import pennylane as qml
@pytest.mark.parametrize('name,expected_output', [('PauliX', -1), ('PauliY', -1), ('PauliZ', 1), ('Hadamard', 0)])
def test_supported_gate_single_wire_no_parameters(self, device, tol, name, expected_output):
    """Tests supported non-parametrized gates that act on a single wire"""
    n_wires = 1
    dev = device(n_wires)
    op = getattr(qml.ops, name)

    @qml.qnode(dev)
    def circuit():
        op(wires=0)
        return qml.expval(qml.Z(0))
    assert np.isclose(circuit(), expected_output, atol=tol(dev.shots))