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
def test_run_sweep_v2_with_unary_rpcs(client):
    setup_run_circuit_with_result_(client, _RESULTS_V2)
    engine = cg.Engine(project_id='proj', context=EngineContext(proto_version=cg.engine.engine.ProtoVersion.V2, enable_streaming=False))
    job = engine.run_sweep(program=_CIRCUIT, job_id='job-id', params=cirq.Points('a', [1, 2]))
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
    assert len(sweeps) == 1
    assert sweeps[0].repetitions == 1
    assert sweeps[0].sweep.single_sweep.points.points == [1, 2]
    client().get_job_async.assert_called_once()
    client().get_job_results_async.assert_called_once()