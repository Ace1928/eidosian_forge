import itertools
import cirq
import cirq_ft
from cirq_ft import infra
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.slow
@pytest.mark.parametrize('num_ones', [*range(5, 9)])
@pytest.mark.parametrize('eps', [0.01])
@allow_deprecated_cirq_ft_use_in_tests
def test_reflection_using_prepare(num_ones, eps):
    data = [1] * num_ones
    prepare_gate = cirq_ft.StatePreparationAliasSampling.from_lcu_probs(data, probability_epsilon=eps)
    gate = cirq_ft.ReflectionUsingPrepare(prepare_gate)
    g, qubit_order, decomposed_circuit = construct_gate_helper_and_qubit_order(gate)
    decomposed_circuit = greedily_allocate_ancilla(decomposed_circuit)
    initial_state_prep = cirq.Circuit(cirq.H.on_each(*g.quregs['selection']))
    initial_state = cirq.dirac_notation(initial_state_prep.final_state_vector())
    assert initial_state == get_3q_uniform_dirac_notation('++++++++')
    result = cirq.Simulator(dtype=np.complex128).simulate(initial_state_prep + decomposed_circuit, qubit_order=qubit_order)
    selection = g.quregs['selection']
    prepared_state = result.final_state_vector.reshape(2 ** len(selection), -1).sum(axis=1)
    signs = '-' * num_ones + '+' * (9 - num_ones)
    assert cirq.dirac_notation(prepared_state) == get_3q_uniform_dirac_notation(signs)