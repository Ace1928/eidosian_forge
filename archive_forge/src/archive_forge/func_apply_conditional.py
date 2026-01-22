from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
@apply_operation.register
def apply_conditional(op: Conditional, state, is_state_batched: bool=False, debugger=None, mid_measurements=None):
    """Applies a conditional operation.

    Args:
        op (Operator): The operation to apply to ``state``
        state (TensorLike): The starting state.
        is_state_batched (bool): Boolean representing whether the state is batched or not
        debugger (_Debugger): The debugger to use
        mid_measurements (dict, None): Mid-circuit measurement dictionary mutated to record the sampled value

    Returns:
        ndarray: output state
    """
    if op.meas_val.concretize(mid_measurements):
        return apply_operation(op.then_op, state, is_state_batched=is_state_batched, debugger=debugger, mid_measurements=mid_measurements)
    return state