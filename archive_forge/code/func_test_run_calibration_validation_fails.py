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
def test_run_calibration_validation_fails():
    engine = cg.Engine(project_id='proj', proto_version=cg.engine.engine.ProtoVersion.V2)
    q1 = cirq.GridQubit(2, 3)
    q2 = cirq.GridQubit(2, 4)
    layer1 = cg.CalibrationLayer('xeb', cirq.Circuit(cirq.CZ(q1, q2)), {'num_layers': 42})
    layer2 = cg.CalibrationLayer('readout', cirq.Circuit(cirq.measure(q1, q2)), {'num_samples': 4242})
    with pytest.raises(ValueError, match='Processor id must be specified'):
        _ = engine.run_calibration(layers=[layer1, layer2], job_id='job-id')
    with pytest.raises(ValueError, match='processor_id and processor_ids'):
        _ = engine.run_calibration(layers=[layer1, layer2], processor_ids=['mysim'], processor_id='mysim', job_id='job-id')