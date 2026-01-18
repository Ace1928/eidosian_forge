from unittest import mock
import datetime
import duet
import pytest
import freezegun
import numpy as np
from google.protobuf.duration_pb2 import Duration
from google.protobuf.text_format import Merge
from google.protobuf.timestamp_pb2 import Timestamp
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.engine import util
from cirq_google.engine.engine import EngineContext
from cirq_google.cloud import quantum
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_reservation_async')
def test_remove_reservation_not_found(get_reservation):
    get_reservation.return_value = None
    processor = cg.EngineProcessor('proj', 'p0', EngineContext(), quantum.QuantumProcessor(schedule_frozen_period=Duration(seconds=10000)))
    with pytest.raises(ValueError):
        processor.remove_reservation('rid')