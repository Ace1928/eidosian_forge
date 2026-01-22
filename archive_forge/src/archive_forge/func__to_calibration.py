import datetime
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union
from google.protobuf import any_pb2
import cirq
from cirq_google.cloud import quantum
from cirq_google.api import v2
from cirq_google.devices import grid_device
from cirq_google.engine import (
def _to_calibration(calibration_any: any_pb2.Any) -> calibration.Calibration:
    metrics = v2.metrics_pb2.MetricsSnapshot.FromString(calibration_any.value)
    return calibration.Calibration(metrics)