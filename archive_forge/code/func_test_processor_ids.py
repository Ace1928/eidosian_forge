import datetime
from unittest import mock
import pytest
import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
def test_processor_ids():
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=quantum.QuantumJob(scheduling_config=quantum.SchedulingConfig(processor_selector=quantum.SchedulingConfig.ProcessorSelector(processor_names=['projects/a/processors/p']))))
    assert job.processor_ids() == ['p']