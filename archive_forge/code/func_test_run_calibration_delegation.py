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
def test_run_calibration_delegation(create_job_async):
    create_job_async.return_value = ('dogs', quantum.QuantumJob())
    program = cg.EngineProgram('woof', 'woof', EngineContext(), result_type=ResultType.Calibration)
    job = program.run_calibration(processor_ids=['lazydog'])
    assert job._job == quantum.QuantumJob()