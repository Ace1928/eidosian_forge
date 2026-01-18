import random
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('selection_bitsize, target_bitsize, n_target_registers, want', [[3, 5, 1, (0, 0)], [2, 2, 3, (16, 86)], [2, 3, 4, (36, 195)], [3, 2, 5, (32, 172)], [4, 1, 10, (36, 189)]])
@allow_deprecated_cirq_ft_use_in_tests
def test_swap_with_zero_t_complexity(selection_bitsize, target_bitsize, n_target_registers, want):
    t_complexity = cirq_ft.TComplexity(t=want[0], clifford=want[1])
    gate = cirq_ft.SwapWithZeroGate(selection_bitsize, target_bitsize, n_target_registers)
    assert t_complexity == cirq_ft.t_complexity(gate)