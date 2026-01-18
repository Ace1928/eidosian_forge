from unittest import mock
import pytest
import cirq
import cirq_google as cg
def test_simulated_backend_with_local_device():
    proc_rec = cg.SimulatedProcessorWithLocalDeviceRecord('rainbow')
    assert isinstance(proc_rec.get_processor(), cg.engine.AbstractProcessor)
    assert proc_rec.processor_id == 'rainbow'
    assert str(proc_rec) == 'rainbow-simulator'
    cirq.testing.assert_equivalent_repr(proc_rec, global_vals={'cirq_google': cg})