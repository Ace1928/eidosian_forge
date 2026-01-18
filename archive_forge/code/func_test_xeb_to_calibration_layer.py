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
def test_xeb_to_calibration_layer():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=0.75, phi=0.0)
    request = XEBPhasedFSimCalibrationRequest(gate=gate, pairs=((q_00, q_01), (q_02, q_03)), options=XEBPhasedFSimCalibrationOptions(n_library_circuits=22, fatol=0.0078125, xatol=0.0078125, fsim_options=XEBPhasedFSimCharacterizationOptions(characterize_theta=True, characterize_zeta=True, characterize_chi=False, characterize_gamma=False, characterize_phi=True)))
    layer = request.to_calibration_layer()
    assert layer == cirq_google.CalibrationLayer(calibration_type='xeb_phased_fsim_characterization', program=cirq.Circuit([gate.on(q_00, q_01), gate.on(q_02, q_03)]), args={'n_library_circuits': 22, 'n_combinations': 10, 'cycle_depths': '5_25_50_100_200_300', 'fatol': 0.0078125, 'xatol': 0.0078125, 'characterize_theta': True, 'characterize_zeta': True, 'characterize_chi': False, 'characterize_gamma': False, 'characterize_phi': True})
    calibration = v2.calibration_pb2.FocusedCalibration()
    new_layer = calibration.layers.add()
    new_layer.calibration_type = layer.calibration_type
    for arg in layer.args:
        arg_to_proto(layer.args[arg], out=new_layer.args[arg])
    cirq_google.CIRCUIT_SERIALIZER.serialize(layer.program, msg=new_layer.layer)
    with open(os.path.dirname(__file__) + '/test_data/xeb_calibration_layer.textproto') as f:
        desired_textproto = f.read()
    layer_str = str(new_layer)
    assert layer_str == desired_textproto