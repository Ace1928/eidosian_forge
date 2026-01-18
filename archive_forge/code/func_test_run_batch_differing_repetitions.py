from unittest import mock
import pytest
import cirq
import cirq_google as cg
from cirq_google.engine.abstract_processor import AbstractProcessor
def test_run_batch_differing_repetitions():
    processor = mock.create_autospec(AbstractProcessor)
    run_name = 'RUN_NAME'
    device_config_name = 'DEVICE_CONFIG_NAME'
    sampler = cg.ProcessorSampler(processor=processor, run_name=run_name, device_config_name=device_config_name)
    job = mock.Mock()
    job.results.return_value = []
    processor.run_sweep.return_value = job
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a))
    circuit2 = cirq.Circuit(cirq.Y(a))
    params1 = [cirq.ParamResolver({'t': 1})]
    params2 = [cirq.ParamResolver({'t': 2})]
    circuits = [circuit1, circuit2]
    params_list = [params1, params2]
    repetitions = [1, 2]
    sampler.run_batch(circuits, params_list, repetitions)
    processor.run_sweep_async.assert_called_with(params=params2, program=circuit2, repetitions=2, run_name=run_name, device_config_name=device_config_name)
    processor.run_batch_async.assert_not_called()