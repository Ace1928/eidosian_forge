from cmath import exp
from math import cos, sin, sqrt
import pytest
import numpy as np
from scipy.linalg import block_diag
from flaky import flaky
import pennylane as qml
@pytest.mark.parametrize('operation', all_ops)
def test_supported_gates_can_be_implemented(self, device_kwargs, operation):
    """Test that the device can implement all its supported gates."""
    device_kwargs['wires'] = 4
    dev = qml.device(**device_kwargs)
    if isinstance(dev, qml.Device):
        if operation not in dev.operations:
            pytest.skip('operation not supported.')
    elif ops[operation].name == 'QubitDensityMatrix':
        prog = dev.preprocess()[0]
        tape = qml.tape.QuantumScript([ops[operation]])
        try:
            prog((tape,))
        except qml.DeviceError:
            pytest.skip('operation not supported on the device')

    @qml.qnode(dev)
    def circuit():
        qml.apply(ops[operation])
        return qml.expval(qml.Identity(wires=0))
    assert isinstance(circuit(), (float, np.ndarray))