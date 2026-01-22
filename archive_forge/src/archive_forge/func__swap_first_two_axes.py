from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def _swap_first_two_axes(grads, first_axis_size, second_axis_size):
    """Transpose the first two axes of an iterable of iterables, returning
    a tuple of tuples."""
    if first_axis_size == 1:
        return tuple((grads[0][i] for i in range(second_axis_size)))
    return tuple((tuple((grads[j][i] for j in range(first_axis_size))) for i in range(second_axis_size)))