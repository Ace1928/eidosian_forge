import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_tensor_index_names():
    qubits = cirq.LineQubit.range(12)
    qubit_map = {qubit: i for i, qubit in enumerate(qubits)}
    state = ccq.mps_simulator.MPSState(qubits=qubit_map, prng=value.parse_random_state(0))
    assert state.i_str(0) == 'i_00'
    assert state.i_str(11) == 'i_11'
    assert state.mu_str(0, 3) == 'mu_0_3'
    assert state.mu_str(3, 0) == 'mu_0_3'