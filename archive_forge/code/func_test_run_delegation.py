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
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.create_job_async')
def test_run_delegation(create_job_async, get_results_async):
    dt = datetime.datetime.now(tz=datetime.timezone.utc)
    create_job_async.return_value = ('steve', quantum.QuantumJob(name='projects/a/programs/b/jobs/steve', execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS), update_time=dt))
    get_results_async.return_value = quantum.QuantumResult(result=util.pack_any(Merge("sweep_results: [{\n        repetitions: 4,\n        parameterized_results: [{\n            params: {\n                assignments: {\n                    key: 'a'\n                    value: 1\n                }\n            },\n            measurement_results: {\n                key: 'q'\n                qubit_measurement_results: [{\n                  qubit: {\n                    id: '1_1'\n                  }\n                  results: '\x06'\n                }]\n            }\n        }]\n    }]\n", v2.result_pb2.Result())))
    program = cg.EngineProgram('a', 'b', EngineContext())
    param_resolver = cirq.ParamResolver({})
    results = program.run(job_id='steve', repetitions=10, param_resolver=param_resolver, processor_ids=['mine'])
    assert results == cg.EngineResult(params=cirq.ParamResolver({'a': 1.0}), measurements={'q': np.array([[False], [True], [True], [False]], dtype=bool)}, job_id='steve', job_finished_time=dt)