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
def _single_meas_grad(result, coeffs, unshifted_coeff, r0):
    """Compute the gradient for a single measurement by taking the linear combination of
    the coefficients and the measurement result.

    If an unshifted term exists, its contribution is added to the gradient.
    """
    if isinstance(result, list) and result == []:
        if unshifted_coeff is None:
            raise ValueError('This gradient component neither has a shifted nor an unshifted component. It should have been identified to have a vanishing gradient earlier on.')
        return qml.math.array(unshifted_coeff * r0)
    result = qml.math.stack(result)
    coeffs = qml.math.convert_like(coeffs, result)
    g = qml.math.tensordot(result, coeffs, [[0], [0]])
    if unshifted_coeff is not None:
        g = g + unshifted_coeff * r0
        g = qml.math.array(g)
    return g