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
def test_supported_languages():
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor())
    assert processor.supported_languages() == []
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor(supported_languages=['lang1', 'lang2']))
    assert processor.supported_languages() == ['lang1', 'lang2']