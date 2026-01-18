import pytest
from flaky import flaky
import pennylane as qml
from pennylane import numpy as pnp  # Import from PennyLane to mirror the standard approach in demos
from pennylane.templates.layers import RandomLayers
def test_pauliz_expectation_analytic(self, device, tol):
    """Test that the tensor product of PauliZ expectation value is correct"""
    n_wires = 2
    dev = device(n_wires)
    dev_def = qml.device('default.qubit', wires=n_wires)
    if dev.name == dev_def.name:
        pytest.skip('Device is default.qubit.')
    supports_tensor = isinstance(dev, qml.devices.Device) or ('supports_tensor_observables' in dev.capabilities() and dev.capabilities()['supports_tensor_observables'])
    if not supports_tensor:
        pytest.skip('Device does not support tensor observables.')
    if dev.shots:
        pytest.skip('Device is in non-analytical mode.')
    theta = 0.432
    phi = 0.123

    def circuit(theta, phi):
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Z(0) @ qml.Z(1))
    qnode_def = qml.QNode(circuit, dev_def)
    qnode = qml.QNode(circuit, dev)
    grad_def = qml.grad(qnode_def, argnum=[0, 1])
    grad = qml.grad(qnode, argnum=[0, 1])
    assert pnp.allclose(qnode(theta, phi), qnode_def(theta, phi), atol=tol(dev.shots))
    assert pnp.allclose(grad(theta, phi), grad_def(theta, phi), atol=tol(dev.shots))