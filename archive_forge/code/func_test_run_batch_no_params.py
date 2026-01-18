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
def test_run_batch_no_params(client):
    setup_run_circuit_with_result_(client, _BATCH_RESULTS_V2)
    engine = cg.Engine(project_id='proj', proto_version=cg.engine.engine.ProtoVersion.V2)
    engine.run_batch(programs=[_CIRCUIT, _CIRCUIT2], job_id='job-id', processor_ids=['mysim'])
    run_context = v2.batch_pb2.BatchRunContext()
    client().create_job_async.call_args[1]['run_context'].Unpack(run_context)
    assert len(run_context.run_contexts) == 2
    for rc in run_context.run_contexts:
        sweeps = rc.parameter_sweeps
        assert len(sweeps) == 1
        assert sweeps[0].repetitions == 1
        assert sweeps[0].sweep == v2.run_context_pb2.Sweep()