import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
class SupportsTComplexityGateWithRegisters(cirq_ft.GateWithRegisters):

    @property
    def signature(self) -> cirq_ft.Signature:
        return cirq_ft.Signature.build(s=1, t=2)

    def _t_complexity_(self) -> cirq_ft.TComplexity:
        return cirq_ft.TComplexity(t=1, clifford=2)