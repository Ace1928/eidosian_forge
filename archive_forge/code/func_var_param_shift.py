from typing import Sequence, Callable
import itertools
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import (
from .finite_difference import finite_diff
from .general_shift_rules import generate_shifted_tapes, process_shifts
from .gradient_transform import _no_trainable_grad
from .parameter_shift import _get_operation_recipe, expval_param_shift
def var_param_shift(tape, dev_wires, argnum=None, shifts=None, gradient_recipes=None, f0=None):
    """Partial derivative using the first-order or second-order parameter-shift rule of a tape
    consisting of a mixture of expectation values and variances of observables.

    Expectation values may be of first- or second-order observables,
    but variances can only be taken of first-order variables.

    .. warning::

        This method can only be executed on devices that support the
        :class:`~.PolyXP` observable.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        dev_wires (.Wires): wires on the device the parameter-shift method is computed on
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivative with respect to all
            trainable indices are returned.
        shifts (list[tuple[int or float]]): List containing tuples of shift values.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple should match the number of frequencies for that parameter.
            If unspecified, equidistant shifts are assumed.
        gradient_recipes (tuple(list[list[float]] or None)): List of gradient recipes
            for the parameter-shift method. One gradient recipe must be provided
            per trainable parameter.
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the gradient recipe contains an unshifted term, this value is used,
            saving a quantum evaluation.

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a
        list of generated tapes, together with a post-processing
        function to be applied to the results of the evaluated tapes
        in order to obtain the Jacobian matrix.
    """
    argnum = argnum or tape.trainable_params
    var_mask = [isinstance(m, VarianceMP) for m in tape.measurements]
    var_idx = np.where(var_mask)[0]
    expval_tape = tape.copy(copy_operations=True)
    for i in var_idx:
        obs = expval_tape.measurements[i].obs
        expval_tape._measurements[i] = qml.expval(op=obs)
    gradient_tapes = [expval_tape]
    pdA_tapes, pdA_fn = expval_param_shift(expval_tape, argnum, shifts, gradient_recipes, f0)
    gradient_tapes.extend(pdA_tapes)
    tape_boundary = len(pdA_tapes) + 1
    expval_sq_tape = tape.copy(copy_operations=True)
    for i in var_idx:
        obs = expval_sq_tape.measurements[i].obs
        A = obs._heisenberg_rep(obs.parameters)
        obs = qml.PolyXP(np.outer(A, A), wires=obs.wires)
        expval_sq_tape._measurements[i] = qml.expval(op=obs)
    pdA2_tapes, pdA2_fn = second_order_param_shift(expval_sq_tape, dev_wires, argnum, shifts, gradient_recipes)
    gradient_tapes.extend(pdA2_tapes)

    def processing_fn(results):
        mask = qml.math.convert_like(qml.math.reshape(var_mask, [-1, 1]), results[0])
        f0 = qml.math.expand_dims(results[0], -1)
        pdA = pdA_fn(results[1:tape_boundary])
        pdA2 = pdA2_fn(results[tape_boundary:])
        pdA = qml.math.stack(pdA)
        return qml.math.where(mask, pdA2 - 2 * f0 * pdA, pdA)
    return (gradient_tapes, processing_fn)