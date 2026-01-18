import cirq
import cirq_ft
from cirq_ft import infra
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('n', [*range(3, 20), 25, 41])
@pytest.mark.parametrize('num_controls', [0, 1])
@allow_deprecated_cirq_ft_use_in_tests
def test_prepare_uniform_superposition(n, num_controls):
    gate = cirq_ft.PrepareUniformSuperposition(n, cv=[1] * num_controls)
    all_qubits = cirq.LineQubit.range(cirq.num_qubits(gate))
    control, target = (all_qubits[:num_controls], all_qubits[num_controls:])
    turn_on_controls = [cirq.X(c) for c in control]
    prepare_uniform_op = gate.on(*control, *target)
    circuit = cirq.Circuit(turn_on_controls, prepare_uniform_op)
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=all_qubits)
    final_target_state = cirq.sub_state_vector(result.final_state_vector, keep_indices=list(range(num_controls, num_controls + len(target))))
    expected_target_state = np.asarray([np.sqrt(1.0 / n)] * n + [0] * (2 ** len(target) - n))
    cirq.testing.assert_allclose_up_to_global_phase(expected_target_state, final_target_state, atol=1e-06)