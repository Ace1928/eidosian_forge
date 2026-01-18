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
def test_xeb_parse_result():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = XEBPhasedFSimCalibrationRequest(gate=gate, pairs=((q_00, q_01), (q_02, q_03)), options=XEBPhasedFSimCalibrationOptions(fsim_options=XEBPhasedFSimCharacterizationOptions(characterize_theta=False, characterize_zeta=False, characterize_chi=False, characterize_gamma=False, characterize_phi=True)))
    result = _load_xeb_results_textproto()
    assert request.parse_result(result) == PhasedFSimCalibrationResult(parameters={(q_00, q_01): PhasedFSimCharacterization(phi=0.0, theta=-0.7853981), (q_02, q_03): PhasedFSimCharacterization(phi=0.0, theta=-0.7853981)}, gate=gate, options=request.options)