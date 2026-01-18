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
def test_run_sweep_params_with_unary_rpcs(client):
    setup_run_circuit_with_result_(client, _RESULTS)
    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=False))
    job = engine.run_sweep(program=_CIRCUIT, params=[cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 2})])
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    client().create_program_async.assert_called_once()
    client().create_job_async.assert_called_once()
    run_context = v2.run_context_pb2.RunContext()
    client().create_job_async.call_args[1]['run_context'].Unpack(run_context)
    sweeps = run_context.parameter_sweeps
    assert len(sweeps) == 2
    for i, v in enumerate([1.0, 2.0]):
        assert sweeps[i].repetitions == 1
        assert sweeps[i].sweep.sweep_function.sweeps[0].single_sweep.points.points == [v]
    client().get_job_async.assert_called_once()
    client().get_job_results_async.assert_called_once()