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
def test_xeb_parse_result_failure():
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = XEBPhasedFSimCalibrationRequest(gate=gate, pairs=(), options=XEBPhasedFSimCalibrationOptions(fsim_options=XEBPhasedFSimCharacterizationOptions(characterize_theta=False, characterize_zeta=False, characterize_chi=False, characterize_gamma=False, characterize_phi=True)))
    result = cirq_google.CalibrationResult(code=cirq_google.api.v2.calibration_pb2.ERROR_CALIBRATION_FAILED, error_message='Test message', token=None, valid_until=None, metrics=cirq_google.Calibration())
    with pytest.raises(PhasedFSimCalibrationError, match='Test message'):
        request.parse_result(result)