import pytest
from flaky import flaky
import pennylane as qml
from pennylane import numpy as pnp  # Import from PennyLane to mirror the standard approach in demos
from pennylane.templates.layers import RandomLayers
@pytest.mark.parametrize('state', [[0, 0], [0, 1], [1, 0], [1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], pnp.array([1, 1, 0, 0]) / pnp.sqrt(2), pnp.array([0, 1, 0, 1]) / pnp.sqrt(2), pnp.array([1, 1, 1, 0]) / pnp.sqrt(3), pnp.array([1, 1, 1, 1]) / 2])
def test_projector_expectation(self, device, state, tol, benchmark):
    """Test that arbitrary multi-mode Projector expectation values are correct"""
    n_wires = 2
    dev = device(n_wires)
    dev_def = qml.device('default.qubit', wires=n_wires)
    if dev.shots:
        pytest.skip('Device is in non-analytical mode.')
    if isinstance(dev, qml.Device) and 'Projector' not in dev.observables:
        pytest.skip('Device does not support the Projector observable.')
    if dev.name == 'default.qubit':
        pytest.skip('Device is default.qubit.')
    theta = 0.432
    phi = 0.123

    def circuit(theta, phi, state):
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Projector(state, wires=[0, 1]))
    qnode_def = qml.QNode(circuit, dev_def)
    qnode = qml.QNode(circuit, dev)
    grad_def = qml.grad(qnode_def, argnum=[0, 1])
    grad = qml.grad(qnode, argnum=[0, 1])

    def workload():
        return (qnode(theta, phi, state), qnode_def(theta, phi, state), grad(theta, phi, state), grad_def(theta, phi, state))
    qnode_res, qnode_def_res, grad_res, grad_def_res = benchmark(workload)
    assert pnp.allclose(qnode_res, qnode_def_res, atol=tol(dev.shots))
    assert pnp.allclose(grad_res, grad_def_res, atol=tol(dev.shots))