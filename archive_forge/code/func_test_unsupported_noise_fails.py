import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_unsupported_noise_fails():
    with pytest.raises(ValueError, match='noise must be unitary or mixture but was'):
        ccq.mps_simulator.MPSSimulator(noise=cirq.amplitude_damp(0.5))