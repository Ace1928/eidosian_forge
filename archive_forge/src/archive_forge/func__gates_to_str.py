from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union
import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str
def _gates_to_str(gates: Iterable[Any], gettr: Callable[[Any], str]=_gate_str) -> str:
    """Converts a list of gates (types/instances) to string by calling gettr (str/repr) on each."""
    return f'[{','.join((gettr(g) for g in gates))}]'