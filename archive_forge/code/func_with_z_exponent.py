from typing import AbstractSet, Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numbers
import numpy as np
import sympy
from cirq import value, ops, protocols, linalg
from cirq.ops import raw_types
from cirq._compat import proper_repr
def with_z_exponent(self, z_exponent: Union[float, sympy.Expr]) -> 'cirq.PhasedXZGate':
    return PhasedXZGate(axis_phase_exponent=self._axis_phase_exponent, x_exponent=self._x_exponent, z_exponent=z_exponent)