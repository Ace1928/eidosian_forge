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
def test_run_circuit_failed_missing_processor_name_with_stream_rpcs(client):
    failed_job = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job-id', execution_status={'state': 'FAILURE', 'failure': {'error_code': 'SYSTEM_ERROR', 'error_message': 'Not good'}})
    stream_future = duet.AwaitableFuture()
    stream_future.try_set_result(failed_job)
    client().run_job_over_stream.return_value = stream_future
    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=True))
    with pytest.raises(RuntimeError, match='Job projects/proj/programs/prog/jobs/job-id on processor UNKNOWN failed. SYSTEM_ERROR: Not good'):
        engine.run(program=_CIRCUIT)