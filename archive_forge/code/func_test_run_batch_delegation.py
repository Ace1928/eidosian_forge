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
def test_run_batch_delegation(create_job_async):
    create_job_async.return_value = ('kittens', quantum.QuantumJob())
    program = cg.EngineProgram('my-meow', 'my-meow', EngineContext(), result_type=ResultType.Batch)
    resolver_list = [cirq.Points('cats', [1.0, 2.0, 3.0]), cirq.Points('cats', [4.0, 5.0, 6.0])]
    job = program.run_batch(job_id='steve', repetitions=10, params_list=resolver_list, processor_ids=['lazykitty'])
    assert job._job == quantum.QuantumJob()