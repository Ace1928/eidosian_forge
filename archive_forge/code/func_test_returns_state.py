import pytest
import pennylane.numpy as pnp
import pennylane as qml
def test_returns_state(self, device_kwargs):
    """Tests that the device reports correctly whether it supports returning the state."""
    device_kwargs['wires'] = 1
    dev = qml.device(**device_kwargs)
    if isinstance(dev, qml.devices.Device):
        pytest.skip('test is old interface specific.')
    cap = dev.capabilities()

    @qml.qnode(dev)
    def circuit():
        qml.X(0)
        return qml.state()
    if not cap.get('returns_state'):
        with pytest.raises(qml.QuantumFunctionError):
            dev.access_state()
        try:
            state = dev.state
        except (AttributeError, NotImplementedError):
            state = None
        assert state is None
    else:
        if dev.shots is not None:
            with pytest.warns(UserWarning, match='Requested state or density matrix with finite shots; the returned'):
                circuit()
        else:
            circuit()
        assert dev.state is not None