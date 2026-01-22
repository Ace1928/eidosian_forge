import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
def _ensure_deterministic_loop_count(self):
    if self.repeat_until or isinstance(self.repetitions, sympy.Expr):
        raise ValueError('Cannot unroll circuit due to nondeterministic repetitions')