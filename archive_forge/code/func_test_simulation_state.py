import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_simulation_state():
    q0, q1 = qubit_order = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(q1, q0))
    mps_simulator = ccq.mps_simulator.MPSSimulator()
    ref_simulator = cirq.Simulator()
    for initial_state in range(4):
        args = mps_simulator._create_simulation_state(initial_state=initial_state, qubits=(q0, q1))
        actual = mps_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=args)
        expected = ref_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
        np.testing.assert_allclose(actual.final_state.to_numpy(), expected.final_state_vector, atol=0.0001)
        assert len(actual.measurements) == 0