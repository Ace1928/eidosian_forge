from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
@apply_operation.register
def apply_mid_measure(op: MidMeasureMP, state, is_state_batched: bool=False, debugger=None, mid_measurements=None):
    """Applies a native mid-circuit measurement.

    Args:
        op (Operator): The operation to apply to ``state``
        state (TensorLike): The starting state.
        is_state_batched (bool): Boolean representing whether the state is batched or not
        debugger (_Debugger): The debugger to use
        mid_measurements (dict, None): Mid-circuit measurement dictionary mutated to record the sampled value

    Returns:
        ndarray: output state
    """
    if is_state_batched:
        raise ValueError('MidMeasureMP cannot be applied to batched states.')
    if not np.allclose(np.linalg.norm(state), 1.0):
        mid_measurements[op] = 0
        return np.zeros_like(state)
    wire = op.wires
    probs = qml.devices.qubit.measure(qml.probs(wire), state)
    sample = np.random.binomial(1, probs[1])
    mid_measurements[op] = sample
    if op.postselect is not None and sample != op.postselect:
        return np.zeros_like(state)
    axis = wire.toarray()[0]
    slices = [slice(None)] * qml.math.ndim(state)
    slices[axis] = int(not sample)
    state[tuple(slices)] = 0.0
    state_norm = np.linalg.norm(state)
    state = state / state_norm
    if op.reset and sample == 1:
        state = apply_operation(qml.X(wire), state, is_state_batched=is_state_batched, debugger=debugger)
    return state