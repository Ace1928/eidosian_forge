import itertools as it
import warnings
from functools import partial
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.measurements import ProbabilityMP, StateMP, VarianceMP
from pennylane.transforms import transform
from .general_shift_rules import (
from .gradient_transform import find_and_validate_gradient_methods
from .parameter_shift import _get_operation_recipe
from .hessian_transform import _process_jacs
def _all_zero_hessian(tape):
    num_params = len(tape.trainable_params)
    zeros_list = []
    for m in tape.measurements:
        shape = 2 ** len(m.wires) if isinstance(m, ProbabilityMP) else ()
        zeros = tuple((tuple((qml.math.zeros(shape) for _ in range(num_params))) for _ in range(num_params)))
        if num_params == 1:
            zeros = zeros[0][0]
        zeros_list.append(zeros)
    if len(tape.measurements) == 1:
        return ([], lambda _: zeros_list[0])
    return ([], lambda _: tuple(zeros_list))