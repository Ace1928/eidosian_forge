import cmath
import math
from typing import AbstractSet, Any, Dict, Optional, Tuple
import numpy as np
import sympy
import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import gate_features, raw_types
def to_exponent(angle_rads: 'cirq.TParamVal') -> 'cirq.TParamVal':
    """Divides angle_rads by symbolic or numerical pi."""
    pi = sympy.pi if protocols.is_parameterized(angle_rads) else np.pi
    return angle_rads / pi