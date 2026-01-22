from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
def _apply_grover_without_matrix(state, op_wires, is_state_batched):
    """Apply GroverOperator to state. This method uses that this operator
    is :math:`2*P-\\mathbb{I}`, where :math:`P` is the projector onto the
    all-plus state. This allows us to compute the new state by replacing summing
    over all axes on which the operation acts, and "filling in" the all-plus state
    in the resulting lower-dimensional state via a Kronecker product.
    """
    num_wires = len(op_wires)
    prefactor = 2 ** (1 - num_wires)
    sum_axes = [w + is_state_batched for w in op_wires]
    collapsed = math.sum(state, axis=tuple(sum_axes))
    if num_wires == len(qml.math.shape(state)) - is_state_batched:
        new_shape = (-1,) + (1,) * num_wires if is_state_batched else (1,) * num_wires
        return prefactor * math.reshape(collapsed, new_shape) - state
    all_plus = math.cast_like(math.full([2] * num_wires, prefactor), state)
    source = list(range(math.ndim(collapsed), math.ndim(state)))
    return math.moveaxis(math.tensordot(collapsed, all_plus, axes=0), source, sum_axes) - state