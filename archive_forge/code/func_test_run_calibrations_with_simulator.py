import itertools
from typing import Optional
from unittest import mock
import numpy as np
import pytest
import cirq
import cirq_google
import cirq_google.calibration.workflow as workflow
import cirq_google.calibration.xeb_wrapper
from cirq.experiments import (
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
def test_run_calibrations_with_simulator():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = SQRT_ISWAP_INV_GATE
    request = FloquetPhasedFSimCalibrationRequest(gate=gate, pairs=((q_00, q_01), (q_02, q_03)), options=FloquetPhasedFSimCalibrationOptions(characterize_theta=True, characterize_zeta=True, characterize_chi=False, characterize_gamma=False, characterize_phi=True))
    simulator = PhasedFSimEngineSimulator.create_with_ideal_sqrt_iswap()
    actual = workflow.run_calibrations([request], simulator)
    assert actual == [PhasedFSimCalibrationResult(parameters={(q_00, q_01): PhasedFSimCharacterization(theta=np.pi / 4, zeta=0.0, chi=None, gamma=None, phi=0.0), (q_02, q_03): PhasedFSimCharacterization(theta=np.pi / 4, zeta=0.0, chi=None, gamma=None, phi=0.0)}, gate=SQRT_ISWAP_INV_GATE, options=FloquetPhasedFSimCalibrationOptions(characterize_theta=True, characterize_zeta=True, characterize_chi=False, characterize_gamma=False, characterize_phi=True))]