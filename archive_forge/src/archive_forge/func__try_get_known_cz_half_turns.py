from typing import Optional, cast, TYPE_CHECKING, Iterable, Tuple, Dict
import sympy
import numpy as np
from cirq import circuits, ops, value, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def _try_get_known_cz_half_turns(op: ops.Operation, no_symbolic: bool=False) -> Optional[value.TParamVal]:
    if not isinstance(op.gate, ops.CZPowGate):
        return None
    h = op.gate.exponent
    if no_symbolic and isinstance(h, sympy.Basic):
        return None
    return h