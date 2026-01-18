from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_parametrized_trace(self, param_backend):
    """
        Continuing from the non-parametrized example, instead of a pure variable
        input, we take a variable that has been multiplied elementwise by a parameter.

        The trace of this expression is then given by

            x11  x21  x12  x22
        [
            [[1   0   0   0]],

            [[0   0   0   0]],

            [[0   0   0   0]],

            [[0   0   0   1]]
        ]
        """
    param_lin_op = linOpHelper((2, 2), type='param', data=2)
    param_backend.param_to_col = {2: 0, -1: 4}
    param_backend.param_to_size = {-1: 1, 2: 4}
    param_backend.var_length = 4
    variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
    var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
    mul_elem_lin_op = linOpHelper(data=param_lin_op)
    param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)
    trace_lin_op = linOpHelper(args=[variable_lin_op])
    out_view = param_backend.trace(trace_lin_op, param_var_view)
    out_repr = out_view.get_tensor_representation(0, 1)
    slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
    expected_idx_zero = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert np.all(slice_idx_zero == expected_idx_zero)
    slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
    expected_idx_one = np.array([[0.0, 0.0, 0.0, 0.0]])
    assert np.all(slice_idx_one == expected_idx_one)
    slice_idx_two = out_repr.get_param_slice(2).toarray()[:, :-1]
    expected_idx_two = np.array([[0.0, 0.0, 0.0, 0.0]])
    assert np.all(slice_idx_two == expected_idx_two)
    slice_idx_three = out_repr.get_param_slice(3).toarray()[:, :-1]
    expected_idx_three = np.array([[0.0, 0.0, 0.0, 1.0]])
    assert np.all(slice_idx_three == expected_idx_three)
    assert out_view.get_tensor_representation(0, 1) == param_var_view.get_tensor_representation(0, 1)