import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_phase_flip_channel():
    d = cirq.phase_flip(0.3)
    np.testing.assert_almost_equal(cirq.kraus(d), (np.sqrt(1.0 - 0.3) * np.eye(2), np.sqrt(0.3) * Z))
    cirq.testing.assert_consistent_channel(d)
    cirq.testing.assert_consistent_mixture(d)