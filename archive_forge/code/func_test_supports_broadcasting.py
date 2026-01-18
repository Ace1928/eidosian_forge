import pytest
import pennylane.numpy as pnp
import pennylane as qml
def test_supports_broadcasting(self, device_kwargs, mocker):
    """Tests that the device reports correctly whether it supports parameter broadcasting
        and that it can execute broadcasted tapes in any case."""
    device_kwargs['wires'] = 1
    dev = qml.device(**device_kwargs)
    if isinstance(dev, qml.devices.Device):
        pytest.skip('test is old interface specific.')
    cap = dev.capabilities()
    assert 'supports_broadcasting' in cap

    @qml.qnode(dev)
    def circuit(x):
        if cap['model'] == 'qubit':
            qml.RX(x, wires=0)
        else:
            qml.Rotation(x, wires=0)
        return qml.probs(wires=0)
    spy = mocker.spy(qml.transforms, 'broadcast_expand')
    circuit(0.5)
    if cap.get('returns_state'):
        orig_shape = pnp.array(dev.access_state()).shape
    spy.assert_not_called()
    x = pnp.array([0.5, 2.1, -0.6], requires_grad=True)
    if cap['supports_broadcasting']:
        res = circuit(x)
        spy.assert_not_called()
        if cap.get('returns_state'):
            assert pnp.array(dev.access_state()).shape != orig_shape
    else:
        res = circuit(x)
        spy.assert_called()
        if cap.get('returns_state'):
            assert pnp.array(dev.access_state()).shape == orig_shape
    assert pnp.ndim(res) == 2
    assert res.shape[0] == 3