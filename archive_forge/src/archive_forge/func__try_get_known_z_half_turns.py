from typing import Optional, cast, TYPE_CHECKING, Iterable, Tuple, Dict
import sympy
import numpy as np
from cirq import circuits, ops, value, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def _try_get_known_z_half_turns(op: ops.Operation, no_symbolic: bool=False) -> Optional[value.TParamVal]:
    g = op.gate
    if isinstance(g, ops.PhasedXZGate) and (not protocols.is_parameterized(g.x_exponent)) and (not protocols.is_parameterized(g.axis_phase_exponent)) and np.isclose(float(g.x_exponent), 0) and np.isclose(float(g.axis_phase_exponent), 0):
        h = g.z_exponent
    elif isinstance(g, ops.ZPowGate):
        h = g.exponent
    else:
        return None
    if no_symbolic and isinstance(h, sympy.Basic):
        return None
    return h