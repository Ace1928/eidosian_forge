from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union
import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str
def _default_name(self) -> str:
    return f'FSimGateFamily: allow_symbol={self.allow_symbols}; atol={self.atol}'