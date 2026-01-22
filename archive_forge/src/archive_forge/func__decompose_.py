import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def _decompose_(self) -> cirq.OP_TREE:
    yield cirq.X(self.qubits[0])