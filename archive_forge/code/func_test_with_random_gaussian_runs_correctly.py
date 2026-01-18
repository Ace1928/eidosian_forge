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
def test_with_random_gaussian_runs_correctly():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([[cirq.X(a), cirq.Y(c)], [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)], [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)], cirq.measure(a, b, c, d, key='z')])
    simulator = cirq.Simulator()
    engine_simulator = PhasedFSimEngineSimulator.create_with_random_gaussian_sqrt_iswap(SQRT_ISWAP_INV_PARAMETERS, simulator=simulator)
    actual = engine_simulator.run(circuit, repetitions=20000).measurements['z']
    expected = simulator.run(circuit, repetitions=20000).measurements['z']
    assert np.allclose(np.average(actual, axis=0), np.average(expected, axis=0), atol=0.1)