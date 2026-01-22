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
def _put_zeros_in_pdA2_involutory(tape, pdA2, involutory_indices):
    """Auxiliary function for replacing parts of the partial derivative of <A^2>
    with the zero gradient of involutory observables.

    Involutory observables in the partial derivative of <A^2> have zero gradients.
    For the pdA2_tapes, we have replaced non-involutory observables with their
    square (A -> A^2). However, involutory observables have been left as-is
    (A), and have not been replaced by their square (A^2 = I). As a result,
    components of the gradient vector will not be correct. We need to replace
    the gradient value with 0 (the known, correct gradient for involutory
    variables).
    """
    new_pdA2 = []
    for i in range(len(tape.measurements)):
        if i in involutory_indices:
            num_params = len(tape.trainable_params)
            item = qml.math.array(0) if num_params == 1 else tuple((qml.math.array(0) for _ in range(num_params)))
        else:
            item = pdA2[i]
        new_pdA2.append(item)
    return tuple(new_pdA2)