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
def test_run_zeta_chi_gamma_calibration_for_moments() -> None:
    parameters_ab = cirq_google.PhasedFSimCharacterization(zeta=0.5, chi=0.4, gamma=0.3)
    parameters_bc = cirq_google.PhasedFSimCharacterization(zeta=-0.5, chi=-0.4, gamma=-0.3)
    parameters_cd = cirq_google.PhasedFSimCharacterization(zeta=0.2, chi=0.3, gamma=0.4)
    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = cirq_google.PhasedFSimEngineSimulator.create_from_dictionary(parameters={(a, b): {SQRT_ISWAP_INV_GATE: parameters_ab.merge_with(SQRT_ISWAP_INV_PARAMETERS)}, (b, c): {cirq_google.ops.SYC: parameters_bc.merge_with(SYCAMORE_PARAMETERS)}, (c, d): {SQRT_ISWAP_INV_GATE: parameters_cd.merge_with(SQRT_ISWAP_INV_PARAMETERS)}})
    circuit = cirq.Circuit([[cirq.X(a), cirq.Y(c)], [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)], [cirq_google.ops.SYC.on(b, c)]])
    options = cirq_google.FloquetPhasedFSimCalibrationOptions(characterize_theta=False, characterize_zeta=True, characterize_chi=True, characterize_gamma=True, characterize_phi=False)
    calibrated_circuit, calibrations = workflow.run_zeta_chi_gamma_compensation_for_moments(circuit, engine_simulator, processor_id=None, options=options)
    assert cirq.allclose_up_to_global_phase(engine_simulator.final_state_vector(calibrated_circuit.circuit), cirq.final_state_vector(circuit))
    assert calibrations == [cirq_google.PhasedFSimCalibrationResult(gate=SQRT_ISWAP_INV_GATE, parameters={(a, b): parameters_ab, (c, d): parameters_cd}, options=options), cirq_google.PhasedFSimCalibrationResult(gate=cirq_google.ops.SYC, parameters={(b, c): parameters_bc}, options=options)]
    assert calibrated_circuit.moment_to_calibration == [None, None, 0, None, None, 1, None]