from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.measurements import VarianceMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .finite_difference import finite_diff
from .general_shift_rules import (
from .gradient_transform import (
def zero_entry(grad_entry):
    """Create a gradient entry that is zero and has the correctly modified shape."""
    new_shape = par_shape + qml.math.shape(grad_entry)[cut_dims:]
    return qml.math.zeros(new_shape, like=grad_entry)