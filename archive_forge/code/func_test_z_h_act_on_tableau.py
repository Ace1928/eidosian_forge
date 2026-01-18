import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_z_h_act_on_tableau():
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.act_on(cirq.Z, ExampleSimulationState(), qubits=())
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.act_on(cirq.H, ExampleSimulationState(), qubits=())
    original_tableau = cirq.CliffordTableau(num_qubits=5, initial_state=31)
    flipped_tableau = cirq.CliffordTableau(num_qubits=5, initial_state=23)
    state = cirq.CliffordTableauSimulationState(tableau=original_tableau.copy(), qubits=cirq.LineQubit.range(5), prng=np.random.RandomState())
    cirq.act_on(cirq.H, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.Z ** 0.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.Z ** 0.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.H, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau
    cirq.act_on(cirq.H, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.Z, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.H, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == original_tableau
    cirq.act_on(cirq.H, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.Z ** 3.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.Z ** 3.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.H, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau
    cirq.act_on(cirq.Z ** 2, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau
    cirq.act_on(cirq.H ** 2, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau
    foo = sympy.Symbol('foo')
    with pytest.raises(TypeError, match='Failed to act action on state'):
        cirq.act_on(cirq.Z ** foo, state, [cirq.LineQubit(1)])
    with pytest.raises(TypeError, match='Failed to act action on state'):
        cirq.act_on(cirq.H ** foo, state, [cirq.LineQubit(1)])
    with pytest.raises(TypeError, match='Failed to act action on state'):
        cirq.act_on(cirq.H ** 1.5, state, [cirq.LineQubit(1)])