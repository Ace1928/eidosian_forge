from typing import Iterable, Optional, Tuple
import collections
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq_google
from cirq_google.calibration.engine_simulator import (
from cirq_google.calibration import (
import cirq
def test_with_random_gaussian_sqrt_iswap_simulates_correctly():
    engine_simulator = PhasedFSimEngineSimulator.create_with_random_gaussian_sqrt_iswap(mean=SQRT_ISWAP_INV_PARAMETERS, sigma=PhasedFSimCharacterization(theta=0.02, zeta=0.05, chi=0.05, gamma=None, phi=0.02))
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([[cirq.X(a), cirq.Y(c)], [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)], [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)], [cirq.FSimGate(np.pi / 4, 0.0).on(b, a), cirq.FSimGate(np.pi / 4, 0.0).on(d, c)]])
    calibrations = engine_simulator.get_calibrations([_create_sqrt_iswap_request([(a, b), (c, d)]), _create_sqrt_iswap_request([(b, c)])])
    parameters = collections.ChainMap(*(calibration.parameters for calibration in calibrations))
    expected_circuit = cirq.Circuit([[cirq.X(a), cirq.X(c)], [cirq.PhasedFSimGate(**parameters[a, b].asdict()).on(a, b), cirq.PhasedFSimGate(**parameters[c, d].asdict()).on(c, d)], [cirq.PhasedFSimGate(**parameters[b, c].asdict()).on(b, c)], [cirq.PhasedFSimGate(**parameters[a, b].asdict()).on(a, b), cirq.PhasedFSimGate(**parameters[c, d].asdict()).on(c, d)]])
    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(expected_circuit)
    assert cirq.allclose_up_to_global_phase(actual, expected, atol=1e-06)