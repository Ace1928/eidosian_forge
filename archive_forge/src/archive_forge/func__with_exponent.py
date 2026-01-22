from typing import List, Tuple
import re
import numpy as np
import pytest
import sympy
import cirq
from cirq import value
from cirq.testing import assert_has_consistent_trace_distance_bound
def _with_exponent(self, exponent):
    return type(self)(self.weight, exponent=exponent, global_shift=self._global_shift)