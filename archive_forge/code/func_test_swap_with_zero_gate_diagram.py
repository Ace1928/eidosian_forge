import random
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_swap_with_zero_gate_diagram():
    gate = cirq_ft.SwapWithZeroGate(3, 2, 4)
    q = cirq.LineQubit.range(cirq.num_qubits(gate))
    circuit = cirq.Circuit(gate.on_registers(**infra.split_qubits(gate.signature, q)))
    cirq.testing.assert_has_diagram(circuit, '\n0: ────@(r⇋0)───\n       │\n1: ────@(r⇋0)───\n       │\n2: ────@(r⇋0)───\n       │\n3: ────swap_0───\n       │\n4: ────swap_0───\n       │\n5: ────swap_1───\n       │\n6: ────swap_1───\n       │\n7: ────swap_2───\n       │\n8: ────swap_2───\n       │\n9: ────swap_3───\n       │\n10: ───swap_3───\n')
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq_ft')