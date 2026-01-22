from typing import List, Tuple
import re
import numpy as np
import pytest
import sympy
import cirq
from cirq import value
from cirq.testing import assert_has_consistent_trace_distance_bound
class WeightedZPowGate(cirq.EigenGate, cirq.testing.SingleQubitGate):

    def __init__(self, weight, **kwargs):
        self.weight = weight
        super().__init__(**kwargs)

    def _value_equality_values_(self):
        return (self.weight, self._canonical_exponent, self._global_shift)
    _value_equality_approximate_values_ = _value_equality_values_

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.diag([1, 0])), (self.weight, np.diag([0, 1]))]

    def _with_exponent(self, exponent):
        return type(self)(self.weight, exponent=exponent, global_shift=self._global_shift)