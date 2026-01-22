import cmath
import math
from typing import AbstractSet, Any, Dict, Optional, Tuple
import numpy as np
import sympy
import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import gate_features, raw_types
def _half_pi_mod_pi(param: 'cirq.TParamVal') -> bool:
    """Returns True iff param, assumed to be in [-pi, pi), is pi/2 (mod pi)."""
    return param in (-np.pi / 2, np.pi / 2, -sympy.pi / 2, sympy.pi / 2)