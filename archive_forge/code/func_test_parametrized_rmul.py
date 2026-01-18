from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_parametrized_rmul(self, param_backend):
    """
        Continuing from the non-parametrized example when the rhs is a parameter,
        instead of multiplying with known values, the matrix is split up into two slices,
        each representing an element of the parameter, i.e. instead of
         x11 x21 x12 x22
        [[1   0   2   0],
         [0   1   0   2]]

         we obtain the list of length two, where we have ones at the entries where previously
         we had the 1 and 2:

         x11  x21  x12  x22
        [
         [[1   0   0   0],
          [0   1   0   0]]

         [[0   0   1   0],
          [0   0   0   1]]
        ]
        """
    variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
    param_backend.var_length = 4
    view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
    view_A = view.get_tensor_representation(0, 4)
    view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
    assert np.all(view_A == np.eye(4))
    rhs_parameter = linOpHelper((2,), type='param', data=2)
    rmul_lin_op = linOpHelper(data=rhs_parameter, args=[variable_lin_op])
    out_view = param_backend.rmul(rmul_lin_op, view)
    out_repr = out_view.get_tensor_representation(0, 2)
    slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
    expected_idx_zero = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    assert np.all(slice_idx_zero == expected_idx_zero)
    slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
    expected_idx_one = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.all(slice_idx_one == expected_idx_one)
    assert out_view.get_tensor_representation(0, 2) == view.get_tensor_representation(0, 2)