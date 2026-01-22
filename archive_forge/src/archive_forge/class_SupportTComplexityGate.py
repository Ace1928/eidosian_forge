import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
class SupportTComplexityGate(cirq.Gate):

    def _num_qubits_(self) -> int:
        return 1

    def _t_complexity_(self) -> cirq_ft.TComplexity:
        return cirq_ft.TComplexity(t=1)