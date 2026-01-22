from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union
import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str
def _approx_eq_or_symbol(self, lhs: Any, rhs: Any) -> bool:
    lhs = lhs if isinstance(lhs, tuple) else (lhs,)
    rhs = rhs if isinstance(rhs, tuple) else (rhs,)
    assert len(lhs) == len(rhs)
    for l, r in zip(lhs, rhs):
        is_parameterized = cirq.is_parameterized(l) or cirq.is_parameterized(r)
        if is_parameterized and (not self.allow_symbols) or (not is_parameterized and (not cirq.approx_eq(l, r, atol=self.atol))):
            return False
    return True