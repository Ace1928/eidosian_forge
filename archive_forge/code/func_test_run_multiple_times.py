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
def test_run_multiple_times(client):
    setup_run_circuit_with_result_(client, _RESULTS)
    engine = cg.Engine(project_id='proj', proto_version=cg.engine.engine.ProtoVersion.V2)
    program = engine.create_program(program=_CIRCUIT)
    program.run(param_resolver=cirq.ParamResolver({'a': 1}))
    run_context = v2.run_context_pb2.RunContext()
    client().create_job_async.call_args[1]['run_context'].Unpack(run_context)
    sweeps1 = run_context.parameter_sweeps
    job2 = program.run_sweep(repetitions=2, params=cirq.Points('a', [3, 4]))
    client().create_job_async.call_args[1]['run_context'].Unpack(run_context)
    sweeps2 = run_context.parameter_sweeps
    results = job2.results()
    assert engine.context.proto_version == cg.engine.engine.ProtoVersion.V2
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    assert len(sweeps1) == 1
    assert sweeps1[0].repetitions == 1
    points1 = sweeps1[0].sweep.sweep_function.sweeps[0].single_sweep.points
    assert points1.points == [1]
    assert len(sweeps2) == 1
    assert sweeps2[0].repetitions == 2
    assert sweeps2[0].sweep.single_sweep.points.points == [3, 4]
    assert client().get_job_async.call_count == 2
    assert client().get_job_results_async.call_count == 2