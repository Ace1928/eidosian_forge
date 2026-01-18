import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_sample_measurement_with_shots(self, device):
    """Test that executing a state measurement with shots raises a warning."""
    dev = device(2)
    _skip_test_for_braket(dev)
    if not dev.shots:
        pytest.skip('If shots=None no warning is raised.')

    class MyMeasurement(StateMeasurement):
        """Dummy state measurement."""

        def process_state(self, state, wire_order):
            return 1

    @qml.qnode(dev)
    def circuit():
        qml.X(0)
        return MyMeasurement()
    if isinstance(dev, qml.Device):
        with pytest.warns(UserWarning, match='Requested measurement MyMeasurement with finite shots'):
            circuit()
    else:
        with pytest.raises(qml.DeviceError):
            circuit()