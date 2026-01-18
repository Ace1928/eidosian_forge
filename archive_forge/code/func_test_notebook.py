import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.infra import split_qubits, merge_qubits, get_named_qubits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.skip(reason='Cirq-FT is deprecated, use Qualtran instead.')
def test_notebook():
    execute_notebook('gate_with_registers')