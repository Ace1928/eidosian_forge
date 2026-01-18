import pytest
import pennylane.numpy as pnp
import pennylane as qml
def test_returns_probs(self, device_kwargs):
    """Tests that the device reports correctly whether it supports reversible differentiation."""
    device_kwargs['wires'] = 1
    dev = qml.device(**device_kwargs)
    if isinstance(dev, qml.devices.Device):
        pytest.skip('test is old interface specific.')
    cap = dev.capabilities()
    if 'returns_probs' not in cap:
        pytest.skip('No returns_probs capability specified by device.')

    @qml.qnode(dev)
    def circuit():
        if cap['model'] == 'qubit':
            qml.X(0)
        else:
            qml.QuadX(wires=0)
        return qml.probs(wires=0)
    if cap['returns_probs']:
        circuit()
    else:
        with pytest.raises(NotImplementedError):
            circuit()