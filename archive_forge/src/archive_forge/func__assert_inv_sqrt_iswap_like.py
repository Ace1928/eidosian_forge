from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
def _assert_inv_sqrt_iswap_like(gate: cirq.Gate):
    assert isinstance(gate, cirq.FSimGate), f'Expected FSimGate, got {gate}'
    if cirq.is_parameterized(gate):
        raise ValueError('Only unparameterized gates are supported. Gate: {gate}.')
    theta = gate.theta
    phi = gate.phi
    assert isinstance(theta, float) and isinstance(phi, float)
    assert np.isclose(theta, np.pi / 4) and np.isclose(phi, 0.0), f'Expected ISWAP ** -0.5 like gate, got {gate}'