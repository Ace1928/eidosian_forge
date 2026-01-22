from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union
import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str
def _get_value_equality_values(self, g: POSSIBLE_FSIM_GATES) -> Any:
    if type(g) == cirq.PhasedISwapPowGate:
        return (g.phase_exponent, *g._iswap._value_equality_values_())
    return g._value_equality_values_()