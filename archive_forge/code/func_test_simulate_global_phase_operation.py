import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_simulate_global_phase_operation():
    q1, q2 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.I(q1), cirq.I(q2), cirq.global_phase_operation(-1j)])
    simulator = cirq.CliffordSimulator()
    result = simulator.simulate(circuit).final_state.state_vector()
    assert np.allclose(result, [-1j, 0, 0, 0])