import datetime
from unittest import mock
import time
import numpy as np
import pytest
import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_circuit_with_unary_rpcs(client):
    setup_run_circuit_with_result_(client, _A_RESULT)
    engine = cg.Engine(project_id='proj', context=EngineContext(service_args={'client_info': 1}, enable_streaming=False))
    result = engine.run(program=_CIRCUIT, program_id='prog', job_id='job-id', processor_ids=['mysim'])
    assert result.repetitions == 1
    assert result.params.param_dict == {'a': 1}
    assert result.measurements == {'q': np.array([[0]], dtype='uint8')}
    client.assert_called_with(service_args={'client_info': 1}, verbose=None)
    client().create_program_async.assert_called_once()
    client().create_job_async.assert_called_once_with(project_id='proj', program_id='prog', job_id='job-id', processor_ids=['mysim'], run_context=util.pack_any(v2.run_context_pb2.RunContext(parameter_sweeps=[v2.run_context_pb2.ParameterSweep(repetitions=1)])), description=None, labels=None, processor_id='', run_name='', device_config_name='')
    client().get_job_async.assert_called_once_with('proj', 'prog', 'job-id', False)
    client().get_job_results_async.assert_called_once_with('proj', 'prog', 'job-id')