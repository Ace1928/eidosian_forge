import random
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('selection_bitsize, target_bitsize, n_target_registers', [[3, 5, 1], [2, 2, 3], [2, 3, 4], [3, 2, 5], [4, 1, 10]])
@allow_deprecated_cirq_ft_use_in_tests
def test_swap_with_zero_gate(selection_bitsize, target_bitsize, n_target_registers):
    gate = cirq_ft.SwapWithZeroGate(selection_bitsize, target_bitsize, n_target_registers)
    all_qubits = cirq.LineQubit.range(cirq.num_qubits(gate))
    selection = all_qubits[:selection_bitsize]
    target = np.array(all_qubits[selection_bitsize:]).reshape((n_target_registers, target_bitsize))
    circuit = cirq.Circuit(gate.on_registers(selection=selection, target=target))
    data = [random.randint(0, 2 ** target_bitsize - 1) for _ in range(n_target_registers)]
    target_state = [int(x) for d in data for x in format(d, f'0{target_bitsize}b')]
    sim = cirq.Simulator(dtype=np.complex128)
    expected_state_vector = np.zeros(2 ** target_bitsize)
    for selection_integer in range(len(data)):
        selection_state = [int(x) for x in format(selection_integer, f'0{selection_bitsize}b')]
        initial_state = selection_state + target_state
        result = sim.simulate(circuit, initial_state=initial_state)
        result_state_vector = cirq.sub_state_vector(result.final_state_vector, keep_indices=list(range(selection_bitsize, selection_bitsize + target_bitsize)))
        expected_state_vector[data[selection_integer]] = 1
        assert cirq.equal_up_to_global_phase(result_state_vector, expected_state_vector)
        expected_state_vector[data[selection_integer]] = 0