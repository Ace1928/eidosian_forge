import cmath
import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.analytical_decompositions.two_qubit_to_cz import (
from cirq.testing import random_two_qubit_circuit_with_czs
def test_trivial_parity_interaction_corner_case():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    nearPi4 = np.pi / 4 * 0.99
    tolerance = 0.01
    circuit = cirq.Circuit(_parity_interaction(q0, q1, -nearPi4, tolerance))
    assert len(circuit) == 2