from math import sqrt, pi
import pytest
import numpy as np
from flaky import flaky
import pennylane as qml
@pytest.mark.parametrize('name,par,expected_output', [('CRX', [0], [-1 / 2, -1 / 2]), ('CRX', [-pi], [-1 / 2, 1]), ('CRX', [pi / 2], [-1 / 2, 1 / 4]), ('CRY', [0], [-1 / 2, -1 / 2]), ('CRY', [-pi], [-1 / 2, 1]), ('CRY', [pi / 2], [-1 / 2, 1 / 4]), ('CRZ', [0], [-1 / 2, -1 / 2]), ('CRZ', [-pi], [-1 / 2, -1 / 2]), ('CRZ', [pi / 2], [-1 / 2, -1 / 2]), ('MultiRZ', [0], [-1 / 2, -1 / 2]), ('MultiRZ', [-pi], [-1 / 2, -1 / 2]), ('MultiRZ', [pi / 2], [-1 / 2, -1 / 2]), ('CRot', [pi / 2, 0, 0], [-1 / 2, -1 / 2]), ('CRot', [0, pi / 2, 0], [-1 / 2, 1 / 4]), ('CRot', [0, 0, pi / 2], [-1 / 2, -1 / 2]), ('CRot', [pi / 2, 0, -pi], [-1 / 2, -1 / 2]), ('CRot', [0, pi / 2, -pi], [-1 / 2, 1 / 4]), ('CRot', [-pi, 0, pi / 2], [-1 / 2, -1 / 2]), ('QubitUnitary', [np.array([[1, 0, 0, 0], [0, 1 / sqrt(2), 1 / sqrt(2), 0], [0, 1 / sqrt(2), -1 / sqrt(2), 0], [0, 0, 0, 1]])], [-1 / 2, -1 / 2]), ('QubitUnitary', [np.array([[-1, 0, 0, 0], [0, 1 / sqrt(2), 1 / sqrt(2), 0], [0, 1 / sqrt(2), -1 / sqrt(2), 0], [0, 0, 0, -1]])], [-1 / 2, -1 / 2])])
def test_supported_gate_two_wires_with_parameters(self, device, tol, name, par, expected_output):
    """Tests supported parametrized gates that act on two wires"""
    n_wires = 2
    dev = device(n_wires)
    op = getattr(qml.ops, name)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(np.array([1 / 2, 0, 0, sqrt(3) / 2]), wires=[0, 1])
        op(*par, wires=[0, 1])
        return (qml.expval(qml.Z(0)), qml.expval(qml.Z(1)))
    assert np.allclose(circuit(), expected_output, atol=tol(dev.shots))