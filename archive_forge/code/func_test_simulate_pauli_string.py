import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_simulate_pauli_string():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit([cirq.PauliString({q: 'X'}), cirq.PauliString({q: 'Z'})])
    simulator = cirq.CliffordSimulator()
    result = simulator.simulate(circuit).final_state.state_vector()
    assert np.allclose(result, [0, -1])