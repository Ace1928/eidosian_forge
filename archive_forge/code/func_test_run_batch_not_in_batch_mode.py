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
def test_run_batch_not_in_batch_mode():
    program = cg.EngineProgram('no-meow', 'no-meow', EngineContext())
    resolver_list = [cirq.Points('cats', [1.0, 2.0, 3.0]), cirq.Points('cats', [4.0, 5.0, 6.0])]
    with pytest.raises(ValueError, match='Can only use run_batch'):
        _ = program.run_batch(repetitions=1, processor_ids=['lazykitty'], params_list=resolver_list)