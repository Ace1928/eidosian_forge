import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_tensor_observables_can_be_implemented(self, device_kwargs):
    """Test that the device can implement a simple tensor observable.
        This test is skipped for devices that do not support tensor observables."""
    device_kwargs['wires'] = 2
    dev = qml.device(**device_kwargs)
    supports_tensor = isinstance(dev, qml.devices.Device) or ('supports_tensor_observables' in dev.capabilities() and dev.capabilities()['supports_tensor_observables'])
    if not supports_tensor:
        pytest.skip('Device does not support tensor observables.')

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(0)
        return qml.expval(qml.Identity(wires=0) @ qml.Identity(wires=1))
    assert isinstance(circuit(), (float, np.ndarray))