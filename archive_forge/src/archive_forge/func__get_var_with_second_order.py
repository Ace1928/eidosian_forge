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
def _get_var_with_second_order(pdA2, f0, pdA):
    """Auxiliary function to compute d(var(A))/dp = d<A^2>/dp -2 * <A> *
    d<A>/dp for the variances.

    The result is converted to an array-like object as some terms (specifically f0) may not be array-like in every case.

    Args:
        pdA (tensor_like[float]): The analytic derivative of <A>.
        pdA2 (tensor_like[float]): The analytic derivatives of the <A^2> observables.
        f0 (tensor_like[float]): Output of the evaluated input tape. If provided,
            and the gradient recipe contains an unshifted term, this value is used,
            saving a quantum evaluation.
    """
    if any((isinstance(term, np.ndarray) for term in [pdA2, f0, pdA])):
        return qml.math.array(pdA2 - 2 * f0 * pdA)
    return pdA2 - 2 * f0 * pdA