from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
class CustomXPowGate(cirq.EigenGate):

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.array([[0.5, 0.5], [0.5, 0.5]])), (1, np.array([[0.5, -0.5], [-0.5, 0.5]]))]

    def __str__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'CustomX'
            return f'CustomX**{self._exponent}'
        return f'CustomXPowGate(exponent={self._exponent}, global_shift={self._global_shift!r})'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.ops.gateset_test.CustomX'
            return f'(cirq.ops.gateset_test.CustomX**{proper_repr(self._exponent)})'
        return f'cirq.ops.gateset_test.CustomXPowGate(exponent={proper_repr(self._exponent)}, global_shift={self._global_shift!r})'

    def _num_qubits_(self) -> int:
        return 1