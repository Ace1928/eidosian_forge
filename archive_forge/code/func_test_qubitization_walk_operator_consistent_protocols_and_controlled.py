import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.algos.generic_select_test import get_1d_Ising_hamiltonian
from cirq_ft.algos.reflection_using_prepare_test import greedily_allocate_ancilla, keep
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_qubitization_walk_operator_consistent_protocols_and_controlled():
    gate = get_walk_operator_for_1d_Ising_model(4, 0.1)
    op = gate.on_registers(**infra.get_named_qubits(gate.signature))
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq\nimport cirq_ft\nimport numpy as np')
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(gate.controlled(), gate.controlled(num_controls=1), gate.controlled(control_values=(1,)), op.controlled_by(cirq.q('control')).gate)
    equals_tester.add_equality_group(gate.controlled(control_values=(0,)), gate.controlled(num_controls=1, control_values=(0,)), op.controlled_by(cirq.q('control'), control_values=(0,)).gate)
    with pytest.raises(NotImplementedError, match='Cannot create a controlled version'):
        _ = gate.controlled(num_controls=2)