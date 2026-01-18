import pytest
import pennylane as qml
def test_tracker_updated_in_execution_mode(self, device):
    """Tests that device update and records during tracking mode"""
    dev = device(1)
    if isinstance(dev, qml.Device) and (not dev.capabilities().get('supports_tracker', False)):
        pytest.skip('Device does not support a tracker')

    @qml.qnode(dev, diff_method='parameter-shift')
    def circ():
        return qml.expval(qml.X(0))
    dev.tracker.active = False
    with dev.tracker:
        circ()
    assert dev.tracker.history['batches'] == [1]
    assert dev.tracker.history['executions'] == [1]