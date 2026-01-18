import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.algos.generic_select_test import get_1d_Ising_lcu_coeffs
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('num_sites, epsilon', [(2, 0.003), pytest.param(3, 0.003, marks=pytest.mark.slow), pytest.param(4, 0.005, marks=pytest.mark.slow), pytest.param(7, 0.008, marks=pytest.mark.slow)])
@allow_deprecated_cirq_ft_use_in_tests
def test_state_preparation_via_coherent_alias_sampling(num_sites, epsilon):
    lcu_coefficients = get_1d_Ising_lcu_coeffs(num_sites)
    gate = cirq_ft.StatePreparationAliasSampling.from_lcu_probs(lcu_probabilities=lcu_coefficients.tolist(), probability_epsilon=epsilon)
    g = cirq_ft.testing.GateHelper(gate)
    qubit_order = g.operation.qubits
    assert len(g.circuit.all_qubits()) < 20
    result = cirq.Simulator(dtype=np.complex128).simulate(g.circuit, qubit_order=qubit_order)
    state_vector = result.final_state_vector
    L, logL = (len(lcu_coefficients), len(g.quregs['selection']))
    state_vector = state_vector.reshape(2 ** logL, len(state_vector) // 2 ** logL)
    num_non_zero = (abs(state_vector) > 1e-06).sum(axis=1)
    prepared_state = state_vector.sum(axis=1)
    assert all(num_non_zero[:L] > 0) and all(num_non_zero[L:] == 0)
    assert all(np.abs(prepared_state[:L]) > 1e-06) and all(np.abs(prepared_state[L:]) <= 1e-06)
    prepared_state = prepared_state[:L] / np.sqrt(num_non_zero[:L])
    np.testing.assert_allclose(lcu_coefficients, abs(prepared_state) ** 2, atol=epsilon)