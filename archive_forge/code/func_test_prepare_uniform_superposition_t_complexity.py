import cirq
import cirq_ft
from cirq_ft import infra
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('n', [*range(3, 41, 3)])
@allow_deprecated_cirq_ft_use_in_tests
def test_prepare_uniform_superposition_t_complexity(n: int):
    gate = cirq_ft.PrepareUniformSuperposition(n)
    result = cirq_ft.t_complexity(gate)
    assert result.rotations <= 2
    assert result.t <= 12 * (n - 1).bit_length()
    gate = cirq_ft.PrepareUniformSuperposition(n, cv=(1,))
    result = cirq_ft.t_complexity(gate)
    assert result.rotations <= 2 + 2 * infra.total_bits(gate.signature)
    assert result.t <= 12 * (n - 1).bit_length()