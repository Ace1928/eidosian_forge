from typing import AbstractSet, Sequence, Union, List, Tuple
import pytest
import numpy as np
import sympy
import cirq
from cirq._compat import proper_repr
from cirq.type_workarounds import NotImplementedType
import cirq.testing.consistent_controlled_gate_op_test as controlled_gate_op_test
class GoodEigenGate(cirq.EigenGate, cirq.testing.SingleQubitGate):

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.diag([1, 0])), (1, np.diag([0, 1]))]

    def __repr__(self):
        return f'GoodEigenGate(exponent={proper_repr(self._exponent)}, global_shift={self._global_shift!r})'