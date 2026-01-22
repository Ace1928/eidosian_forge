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
def _generate_offdiag_tapes(tape, idx, first_order_recipes, add_unshifted, tapes, coeffs):
    """Combine two univariate first order recipes and create
    multi-shifted tapes to compute the off-diagonal entry of the Hessian."""
    recipe_i = first_order_recipes[idx[0]]
    recipe_j = first_order_recipes[idx[1]]
    combined_rules = _combine_shift_rules([recipe_i, recipe_j])
    if np.allclose(combined_rules[0, 1:3], 1.0) and np.allclose(combined_rules[0, 3:5], 0.0):
        if add_unshifted:
            tapes.insert(0, tape)
            add_unshifted = False
        unshifted_coeff = combined_rules[0, 0]
        combined_rules = combined_rules[1:]
    else:
        unshifted_coeff = None
    s = combined_rules[:, 3:5]
    m = combined_rules[:, 1:3]
    new_tapes = generate_multishifted_tapes(tape, idx, s, m)
    tapes.extend(new_tapes)
    coeffs.append(combined_rules[:, 0])
    return (add_unshifted, unshifted_coeff)