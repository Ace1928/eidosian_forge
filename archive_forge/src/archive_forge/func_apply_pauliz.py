from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
@apply_operation.register
def apply_pauliz(op: qml.Z, state, is_state_batched: bool=False, debugger=None, **_):
    """Apply pauliz to state."""
    axis = op.wires[0] + is_state_batched
    n_dim = math.ndim(state)
    if n_dim >= 9 and math.get_interface(state) == 'tensorflow':
        return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)
    sl_0 = _get_slice(0, axis, n_dim)
    sl_1 = _get_slice(1, axis, n_dim)
    state1 = math.multiply(state[sl_1], -1)
    return math.stack([state[sl_0], state1], axis=axis)