import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.infra import split_qubits, merge_qubits, get_named_qubits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_selection_registers_consistent():
    with pytest.raises(ValueError, match='iteration length must be in '):
        _ = cirq_ft.SelectionRegister('a', 3, 10)
    with pytest.raises(ValueError, match='should be flat'):
        _ = cirq_ft.SelectionRegister('a', bitsize=1, shape=(3, 5), iteration_length=5)
    selection_reg = cirq_ft.Signature([cirq_ft.SelectionRegister('n', bitsize=3, iteration_length=5), cirq_ft.SelectionRegister('m', bitsize=4, iteration_length=12)])
    assert selection_reg[0] == cirq_ft.SelectionRegister('n', 3, 5)
    assert selection_reg[1] == cirq_ft.SelectionRegister('m', 4, 12)
    assert selection_reg[:1] == tuple([cirq_ft.SelectionRegister('n', 3, 5)])