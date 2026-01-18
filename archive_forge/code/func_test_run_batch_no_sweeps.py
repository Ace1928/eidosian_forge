import datetime
from unittest import mock
import pytest
import numpy as np
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
from cirq_google.engine.result_type import ResultType
@mock.patch('cirq_google.engine.engine_client.EngineClient.create_job_async')
def test_run_batch_no_sweeps(create_job_async):
    create_job_async.return_value = ('kittens', quantum.QuantumJob())
    program = cg.EngineProgram('my-meow', 'my-meow', _program=quantum.QuantumProgram(code=_BATCH_PROGRAM_V2), context=EngineContext(), result_type=ResultType.Batch)
    job = program.run_batch(job_id='steve', repetitions=10, processor_ids=['lazykitty'])
    assert job._job == quantum.QuantumJob()
    batch_run_context = v2.batch_pb2.BatchRunContext()
    create_job_async.call_args[1]['run_context'].Unpack(batch_run_context)
    assert len(batch_run_context.run_contexts) == 1