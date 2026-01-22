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
def _single_shot_batch_grad(unsupported_grads, supported_grads):
    """Auxiliary function for post-processing one batch of supported and unsupported gradients corresponding to
            finite shot execution.

            If the device used a shot vector, gradients corresponding to a single component of the shot vector should be
            passed to this aux function.
            """
    multi_measure = len(tape.measurements) > 1
    if not multi_measure:
        res = []
        for i, j in zip(unsupported_grads, supported_grads):
            component = qml.math.array(i + j)
            res.append(component)
        return tuple(res)
    combined_grad = []
    for meas_res1, meas_res2 in zip(unsupported_grads, supported_grads):
        meas_grad = []
        for param_res1, param_res2 in zip(meas_res1, meas_res2):
            component = qml.math.array(param_res1 + param_res2)
            meas_grad.append(component)
        meas_grad = tuple(meas_grad)
        combined_grad.append(meas_grad)
    return tuple(combined_grad)