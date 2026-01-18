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
def test_run_batch_no_processors():
    program = cg.EngineProgram('no-meow', 'no-meow', EngineContext(), result_type=ResultType.Batch)
    resolver_list = [cirq.Points('cats', [1.0, 2.0]), cirq.Points('cats', [3.0, 4.0])]
    with pytest.raises(ValueError, match='No processors specified'):
        _ = program.run_batch(repetitions=1, params_list=resolver_list)