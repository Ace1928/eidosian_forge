import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.infra import split_qubits, merge_qubits, get_named_qubits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_registers_build():
    regs1 = cirq_ft.Signature([cirq_ft.Register('r1', 5), cirq_ft.Register('r2', 2)])
    regs2 = cirq_ft.Signature.build(r1=5, r2=2)
    assert regs1 == regs2