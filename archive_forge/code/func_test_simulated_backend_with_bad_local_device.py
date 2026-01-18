from unittest import mock
import pytest
import cirq
import cirq_google as cg
def test_simulated_backend_with_bad_local_device():
    proc_rec = cg.SimulatedProcessorWithLocalDeviceRecord('my_processor')
    with pytest.raises(KeyError):
        proc_rec.get_device()