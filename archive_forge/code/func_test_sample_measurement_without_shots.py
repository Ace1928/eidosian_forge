import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_sample_measurement_without_shots(self, device):
    """Test that executing a sampled measurement with ``shots=None`` raises an error."""
    dev = device(2)
    if dev.shots:
        pytest.skip('If shots!=None no error is raised.')

    class MyMeasurement(SampleMeasurement):
        """Dummy sampled measurement."""

        def process_samples(self, samples, wire_order, shot_range=None, bin_size=None):
            return 1

    @qml.qnode(dev)
    def circuit():
        qml.X(0)
        return (MyMeasurement(wires=[0]), MyMeasurement(wires=[1]))
    with pytest.raises((ValueError, qml.DeviceError)):
        circuit()