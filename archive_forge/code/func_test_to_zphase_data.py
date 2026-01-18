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
def test_to_zphase_data():
    q0, q1, q2 = cirq.GridQubit.rect(1, 3)
    result_1 = PhasedFSimCalibrationResult({(q0, q1): PhasedFSimCharacterization(zeta=0.1, gamma=0.2), (q1, q2): PhasedFSimCharacterization(zeta=0.3, gamma=0.4)}, gate=cirq_google.SycamoreGate(), options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION)
    result_2 = PhasedFSimCalibrationResult({(q0, q1): PhasedFSimCharacterization(zeta=0.5, gamma=0.6), (q1, q2): PhasedFSimCharacterization(zeta=0.7, gamma=0.8)}, gate=cirq.ISwapPowGate(), options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION)
    assert to_zphase_data([result_1, result_2]) == {'syc': {'zeta': {(q0, q1): 0.1, (q1, q2): 0.3}, 'gamma': {(q0, q1): 0.2, (q1, q2): 0.4}}, 'sqrt_iswap': {'zeta': {(q0, q1): 0.5, (q1, q2): 0.7}, 'gamma': {(q0, q1): 0.6, (q1, q2): 0.8}}}
    result_3 = PhasedFSimCalibrationResult({(q0, q1): PhasedFSimCharacterization(theta=0.01), (q1, q2): PhasedFSimCharacterization(zeta=0.02), (q2, q0): PhasedFSimCharacterization(zeta=0.03, gamma=0.04, theta=0.05)}, gate=cirq_google.SycamoreGate(), options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION)
    assert to_zphase_data([result_1, result_3]) == {'syc': {'zeta': {(q0, q1): 0.1, (q1, q2): 0.02, (q2, q0): 0.03}, 'gamma': {(q0, q1): 0.2, (q1, q2): 0.4, (q2, q0): 0.04}, 'theta': {(q0, q1): 0.01, (q2, q0): 0.05}}}