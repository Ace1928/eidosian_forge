import os
from typing import cast
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import sympy
from google.protobuf import text_format
import cirq
import cirq_google
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.calibration.phased_fsim import (
from cirq_google.serialization.arg_func_langs import arg_to_proto
def _load_xeb_results_textproto() -> cirq_google.CalibrationResult:
    with open(os.path.dirname(__file__) + '/test_data/xeb_results.textproto') as f:
        metrics_snapshot = text_format.Parse(f.read(), cirq_google.api.v2.metrics_pb2.MetricsSnapshot())
    return cirq_google.CalibrationResult(code=cirq_google.api.v2.calibration_pb2.SUCCESS, error_message=None, token=None, valid_until=None, metrics=cirq_google.Calibration(metrics_snapshot))