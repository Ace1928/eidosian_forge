from typing import AbstractSet, Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numbers
import numpy as np
import sympy
from cirq import value, ops, protocols, linalg
from cirq.ops import raw_types
from cirq._compat import proper_repr
@property
def z_exponent(self) -> Union[float, sympy.Expr]:
    return self._z_exponent