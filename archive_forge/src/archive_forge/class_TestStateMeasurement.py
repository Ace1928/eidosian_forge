import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
class TestStateMeasurement:
    """Tests for the SampleMeasurement class."""

    def test_custom_state_measurement(self, device):
        """Test the execution of a custom state measurement."""
        dev = device(2)
        _skip_test_for_braket(dev)
        if dev.shots:
            pytest.skip("Some plugins don't update state information when shots is not None.")

        class MyMeasurement(StateMeasurement):
            """Dummy state measurement."""

            def process_state(self, state, wire_order):
                return 1

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return MyMeasurement()
        assert circuit() == 1

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

    def test_method_overriden_by_device(self, device):
        """Test that the device can override a measurement process."""
        dev = device(2)
        _skip_test_for_braket(dev)
        if isinstance(dev, qml.devices.Device):
            pytest.skip('test is specific to old device interface')

        @qml.qnode(dev, interface='autograd')
        def circuit():
            qml.X(0)
            return qml.state()
        circuit.device.measurement_map[StateMP] = 'test_method'
        circuit.device.test_method = lambda obs, shot_range=None, bin_size=None: 2
        assert circuit() == 2