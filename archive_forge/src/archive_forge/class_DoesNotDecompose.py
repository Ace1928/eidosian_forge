import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
class DoesNotDecompose(cirq.Operation):

    def _t_complexity_(self) -> cirq_ft.TComplexity:
        return cirq_ft.TComplexity(t=1, clifford=2, rotations=3)

    @property
    def qubits(self):
        return []

    def with_qubits(self, _):
        pass