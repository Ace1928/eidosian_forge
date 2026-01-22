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
def _evaluate_gradient(tape, res, data, r0):
    """Use shifted tape evaluations and parameter-shift rule coefficients to evaluate
    a gradient result. If res is an empty list, ``r0`` and ``data[3]``, which is the
    coefficient for the unshifted term, must be given and not None.
    """
    _, coeffs, fn, unshifted_coeff, _ = data
    if fn is not None:
        res = fn(res)
    num_measurements = len(tape.measurements)
    if num_measurements == 1:
        if not tape.shots.has_partitioned_shots:
            return _single_meas_grad(res, coeffs, unshifted_coeff, r0)
        g = []
        len_shot_vec = tape.shots.num_copies
        if r0 is None:
            r0 = [None] * int(len_shot_vec)
        for i in range(len_shot_vec):
            shot_comp_res = [r[i] for r in res]
            shot_comp_res = _single_meas_grad(shot_comp_res, coeffs, unshifted_coeff, r0[i])
            g.append(shot_comp_res)
        return tuple(g)
    g = []
    if not tape.shots.has_partitioned_shots:
        return _multi_meas_grad(res, coeffs, r0, unshifted_coeff, num_measurements)
    for idx_shot_comp in range(tape.shots.num_copies):
        single_shot_component_result = [result_for_each_param[idx_shot_comp] for result_for_each_param in res]
        multi_meas_grad = _multi_meas_grad(single_shot_component_result, coeffs, r0, unshifted_coeff, num_measurements)
        g.append(multi_meas_grad)
    return tuple(g)