from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def angle_to_exponent_key(t: Union[float, sympy.Basic]) -> Optional[str]:
    if isinstance(t, sympy.Basic):
        if t == sympy.Symbol('t'):
            return '^t'
        if t == -sympy.Symbol('t'):
            return '^-t'
        return None
    if same_half_turns(t, 1):
        return ''
    if same_half_turns(t, 0.5):
        return '^½'
    if same_half_turns(t, -0.5):
        return '^-½'
    if same_half_turns(t, 0.25):
        return '^¼'
    if same_half_turns(t, -0.25):
        return '^-¼'
    return None