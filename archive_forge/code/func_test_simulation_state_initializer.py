import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_simulation_state_initializer():
    expected_classical_data = cirq.ClassicalDataDictionaryStore(_records={cirq.MeasurementKey('test'): [(4,)]})
    s = ccq.mps_simulator.MPSState(qubits=(cirq.LineQubit(0),), prng=np.random.RandomState(0), classical_data=expected_classical_data)
    assert s.qubits == (cirq.LineQubit(0),)
    assert s.classical_data == expected_classical_data