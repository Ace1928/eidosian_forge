import cirq
import cirq_ft
from cirq_ft import infra
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_prepare_uniform_superposition_consistent_protocols():
    gate = cirq_ft.PrepareUniformSuperposition(5, cv=(1, 0))
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq_ft')
    expected_symbols = ('@', '@(0)', 'UNIFORM(5)', 'target', 'target')
    assert cirq.circuit_diagram_info(gate).wire_symbols == expected_symbols
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(cirq_ft.PrepareUniformSuperposition(5, cv=(1, 0)), cirq_ft.PrepareUniformSuperposition(5, cv=[1, 0]))
    equals_tester.add_equality_group(cirq_ft.PrepareUniformSuperposition(5, cv=(0, 1)), cirq_ft.PrepareUniformSuperposition(5, cv=[0, 1]))
    equals_tester.add_equality_group(cirq_ft.PrepareUniformSuperposition(5), cirq_ft.PrepareUniformSuperposition(5, cv=()), cirq_ft.PrepareUniformSuperposition(5, cv=[]))