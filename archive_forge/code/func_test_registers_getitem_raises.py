import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.infra import split_qubits, merge_qubits, get_named_qubits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_registers_getitem_raises():
    g = cirq_ft.Signature.build(a=4, b=3, c=2)
    with pytest.raises(TypeError, match='indices must be integers or slices'):
        _ = g[2.5]
    selection_reg = cirq_ft.Signature([cirq_ft.SelectionRegister('n', bitsize=3, iteration_length=5)])
    with pytest.raises(TypeError, match='indices must be integers or slices'):
        _ = selection_reg[2.5]