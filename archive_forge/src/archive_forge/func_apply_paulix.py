from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
@apply_operation.register
def apply_paulix(op: qml.X, state, is_state_batched: bool=False, debugger=None, **_):
    """Apply :class:`pennylane.PauliX` operator to the quantum state"""
    axis = op.wires[0] + is_state_batched
    return math.roll(state, 1, axis)