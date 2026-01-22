from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
class TestParametrizedBackends:

    @staticmethod
    @pytest.fixture(params=backends)
    def param_backend(request):
        kwargs = {'id_to_col': {1: 0}, 'param_to_size': {-1: 1, 2: 2}, 'param_to_col': {2: 0, -1: 2}, 'param_size_plus_one': 3, 'var_length': 2}
        backend = CanonBackend.get_backend(request.param, **kwargs)
        assert isinstance(backend, PythonCanonBackend)
        return backend

    def test_parametrized_diag_vec(self, param_backend):
        """
        Starting with a parametrized expression
        x1  x2
        [[[1  0],
         [0  0]],

         [[0  0],
         [0  1]]]

        diag_vec(x) means we introduce zero rows as if the vector was the diagonal
        of an n x n matrix, with n the length of x.

        Thus, when using the same columns as before, we now have

         x1  x2
        [[[1  0],
          [0  0],
          [0  0],
          [0  0]]

         [[0  0],
          [0  0],
          [0  0],
          [0  1]]]
        """
        param_lin_op = linOpHelper((2,), type='param', data=2)
        param_backend.param_to_col = {2: 0, -1: 3}
        variable_lin_op = linOpHelper((2,), type='variable', data=1)
        var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        mul_elem_lin_op = linOpHelper(data=param_lin_op)
        param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)
        diag_vec_lin_op = linOpHelper(shape=(2, 2), data=0)
        out_view = param_backend.diag_vec(diag_vec_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 4)
        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        assert np.all(slice_idx_zero == expected_idx_zero)
        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        assert np.all(slice_idx_one == expected_idx_one)
        assert out_view.get_tensor_representation(0, 4) == param_var_view.get_tensor_representation(0, 4)

    def test_parametrized_diag_vec_with_offset(self, param_backend):
        """
        Starting with a parametrized expression
        x1  x2
        [[[1  0],
          [0  0]],

         [[0  0],
          [0  1]]]

        diag_vec(x, k) means we introduce zero rows as if the vector was the k-diagonal
        of an n+|k| x n+|k| matrix, with n the length of x.

        Thus, for k=1 and using the same columns as before, want to represent
        [[0  x1 0],
         [0  0  x2],
         [0  0  0]]
        parametrized across two slices, i.e., unrolled in column-major order:

        slice 0         slice 1
         x1  x2          x1  x2
        [[0  0],        [[0  0],
         [0  0],         [0  0],
         [0  0],         [0  0],
         [1  0],         [0  0],
         [0  0],         [0  0],
         [0  0],         [0  0],
         [0  0],         [0  0],
         [0  0],         [0  1],
         [0  0]]         [0  0]]
        """
        param_lin_op = linOpHelper((2,), type='param', data=2)
        param_backend.param_to_col = {2: 0, -1: 3}
        variable_lin_op = linOpHelper((2,), type='variable', data=1)
        var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        mul_elem_lin_op = linOpHelper(data=param_lin_op)
        param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)
        k = 1
        diag_vec_lin_op = linOpHelper(shape=(3, 3), data=k)
        out_view = param_backend.diag_vec(diag_vec_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 9)
        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        assert np.all(slice_idx_zero == expected_idx_zero)
        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        assert np.all(slice_idx_one == expected_idx_one)
        assert out_view.get_tensor_representation(0, 9) == param_var_view.get_tensor_representation(0, 9)

    def test_parametrized_sum_entries(self, param_backend):
        """
        starting with a parametrized expression
        x1  x2
        [[[1  0],
         [0  0]],

         [[0  0],
         [0  1]]]

        sum_entries(x) means we consider the entries in all rows, i.e., we sum along the row axis.

        Thus, when using the same columns as before, we now have

         x1  x2
        [[[1  0]],

         [[0  1]]]
        """
        param_lin_op = linOpHelper((2,), type='param', data=2)
        param_backend.param_to_col = {2: 0, -1: 3}
        variable_lin_op = linOpHelper((2,), type='variable', data=1)
        var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        mul_elem_lin_op = linOpHelper(data=param_lin_op)
        param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)
        sum_entries_lin_op = linOpHelper()
        out_view = param_backend.sum_entries(sum_entries_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 1)
        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1.0, 0.0]])
        assert np.all(slice_idx_zero == expected_idx_zero)
        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0.0, 1.0]])
        assert np.all(slice_idx_one == expected_idx_one)
        assert out_view.get_tensor_representation(0, 1) == param_var_view.get_tensor_representation(0, 1)

    def test_parametrized_mul(self, param_backend):
        """
        Continuing from the non-parametrized example when the lhs is a parameter,
        instead of multiplying with known values, the matrix is split up into four slices,
        each representing an element of the parameter, i.e. instead of
         x11 x21 x12 x22
        [[1   2   0   0],
         [3   4   0   0],
         [0   0   1   2],
         [0   0   3   4]]

         we obtain the list of length four, where we have ones at the entries where previously
         we had the 1, 3, 2, and 4 (again flattened in column-major order):

            x11  x21  x12  x22
        [
            [[1   0   0   0],
             [0   0   0   0],
             [0   0   1   0],
             [0   0   0   0]],

            [[0   0   0   0],
             [1   0   0   0],
             [0   0   0   0],
             [0   0   1   0]],

            [[0   1   0   0],
             [0   0   0   0],
             [0   0   0   1],
             [0   0   0   0]],

            [[0   0   0   0],
             [0   1   0   0],
             [0   0   0   0],
             [0   0   0   1]]
        ]
        """
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        param_backend.param_to_size = {-1: 1, 2: 4}
        param_backend.param_to_col = {2: 0, -1: 4}
        param_backend.param_size_plus_one = 5
        param_backend.var_length = 4
        view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        lhs_parameter = linOpHelper((2, 2), type='param', data=2)
        mul_lin_op = linOpHelper(data=lhs_parameter, args=[variable_lin_op])
        out_view = param_backend.mul(mul_lin_op, view)
        out_repr = out_view.get_tensor_representation(0, 4)
        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        assert np.all(slice_idx_zero == expected_idx_zero)
        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        assert np.all(slice_idx_one == expected_idx_one)
        slice_idx_two = out_repr.get_param_slice(2).toarray()[:, :-1]
        expected_idx_two = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
        assert np.all(slice_idx_two == expected_idx_two)
        slice_idx_three = out_repr.get_param_slice(3).toarray()[:, :-1]
        expected_idx_three = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        assert np.all(slice_idx_three == expected_idx_three)
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_parametrized_rhs_mul(self, param_backend):
        """
        Continuing from the non-parametrized example when the expression
        that is multiplied by is parametrized. For a variable that
        was multiplied elementwise by a parameter, instead of
         x11 x21 x12 x22
        [[1   2   0   0],
         [3   4   0   0],
         [0   0   1   2],
         [0   0   3   4]]

         we obtain the list of length four, where we have the same entries as before
         but each variable maps to a different parameter slice:

            x11  x21  x12  x22
        [
            [[1   0   0   0],
             [3   0   0   0],
             [0   0   0   0],
             [0   0   0   0]],

            [[0   2   0   0],
             [0   4   0   0],
             [0   0   0   0],
             [0   0   0   0]],

            [[0   0   0   0],
             [0   0   0   0],
             [0   0   1   0],
             [0   0   3   0]],

            [[0   0   0   0],
             [0   0   0   0],
             [0   0   0   2],
             [0   0   0   4]]
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
        lhs = linOpHelper((2, 2), type='dense_const', data=np.array([[1, 2], [3, 4]]))
        mul_lin_op = linOpHelper(data=lhs, args=[variable_lin_op])
        out_view = param_backend.mul(mul_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 4)
        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        assert np.all(slice_idx_zero == expected_idx_zero)
        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0.0, 2.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        assert np.all(slice_idx_one == expected_idx_one)
        slice_idx_two = out_repr.get_param_slice(2).toarray()[:, :-1]
        expected_idx_two = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 3.0, 0.0]])
        assert np.all(slice_idx_two == expected_idx_two)
        slice_idx_three = out_repr.get_param_slice(3).toarray()[:, :-1]
        expected_idx_three = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 4.0]])
        assert np.all(slice_idx_three == expected_idx_three)
        assert out_view.get_tensor_representation(0, 4) == param_var_view.get_tensor_representation(0, 4)

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

    def test_parametrized_rhs_rmul(self, param_backend):
        """
        Continuing from the non-parametrized example when the expression
        that is multiplied by is parametrized. For a variable that
        was multiplied elementwise by a parameter, instead of

         x11 x21 x12 x22
        [[1   0   3   0],
         [0   1   0   3],
         [2   0   4   0],
         [0   2   0   4]]

         we obtain the list of length four, where we have the same entries as before
         but each variable maps to a different parameter slice:

         x11  x21  x12  x22
        [
         [[1   0   0   0],
          [0   0   0   0],
          [2   0   0   0],
          [0   0   0   0]]

         [[0   0   0   0],
          [0   1   0   0],
          [0   0   0   0],
          [0   2   0   0]]

         [[0   0   3   0],
          [0   0   0   0],
          [0   0   4   0],
          [0   0   0   0]]

         [[0   0   0   0],
          [0   0   0   3],
          [0   0   0   0],
          [0   0   0   4]]
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
        rhs = linOpHelper((2, 2), type='dense_const', data=np.array([[1, 2], [3, 4]]))
        rmul_lin_op = linOpHelper(data=rhs, args=[variable_lin_op])
        out_view = param_backend.rmul(rmul_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 4)
        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        assert np.all(slice_idx_zero == expected_idx_zero)
        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
        assert np.all(slice_idx_one == expected_idx_one)
        slice_idx_two = out_repr.get_param_slice(2).toarray()[:, :-1]
        expected_idx_two = np.array([[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        assert np.all(slice_idx_two == expected_idx_two)
        slice_idx_three = out_repr.get_param_slice(3).toarray()[:, :-1]
        expected_idx_three = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]])
        assert np.all(slice_idx_three == expected_idx_three)
        assert out_view.get_tensor_representation(0, 4) == param_var_view.get_tensor_representation(0, 4)

    def test_mul_elementwise_parametrized(self, param_backend):
        """
        Continuing the non-parametrized example when 'a' is a parameter, instead of multiplying
        with known values, the matrix is split up into two slices, each representing an element
        of the parameter, i.e. instead of
         x1  x2
        [[2  0],
         [0  3]]

         we obtain the list of length two:

          x1  x2
        [
         [[1  0],
          [0  0]],

         [[0  0],
          [0  1]]
        ]
        """
        variable_lin_op = linOpHelper((2,), type='variable', data=1)
        view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 2)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))
        lhs_parameter = linOpHelper((2,), type='param', data=2)
        mul_elementwise_lin_op = linOpHelper(data=lhs_parameter)
        out_view = param_backend.mul_elem(mul_elementwise_lin_op, view)
        out_repr = out_view.get_tensor_representation(0, 2)
        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1, 0], [0, 0]])
        assert np.all(slice_idx_zero == expected_idx_zero)
        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0, 0], [0, 1]])
        assert np.all(slice_idx_one == expected_idx_one)
        assert out_view.get_tensor_representation(0, 2) == view.get_tensor_representation(0, 2)

    def test_parametrized_div(self, param_backend):
        """
        Continuing from the non-parametrized example when the expression
        that is divided by is parametrized. For a variable that
        was multiplied elementwise by a parameter, instead of
         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1/3 0   0],
         [0   0   1/2 0],
         [0   0   0   1/4]]

         we obtain the list of length four, where we have the quotients at the same entries
         but each variable maps to a different parameter slice:

            x11  x21  x12  x22
        [
            [[1   0   0   0],
             [0   0   0   0],
             [0   0   0   0],
             [0   0   0   0]],

            [[0   0   0   0],
             [0   1/3 0   0],
             [0   0   0   0],
             [0   0   0   0]],

            [[0   0   0   0],
             [0   0   0   0],
             [0   0   1/2 0],
             [0   0   0   0]],

            [[0   0   0   0],
             [0   0   0   0],
             [0   0   0   0],
             [0   0   0   1/4]]
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
        lhs = linOpHelper((2, 2), type='dense_const', data=np.array([[1, 2], [3, 4]]))
        div_lin_op = linOpHelper(data=lhs)
        out_view = param_backend.div(div_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 4)
        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        assert np.all(slice_idx_zero == expected_idx_zero)
        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1 / 3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        assert np.all(slice_idx_one == expected_idx_one)
        slice_idx_two = out_repr.get_param_slice(2).toarray()[:, :-1]
        expected_idx_two = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1 / 2, 0.0], [0.0, 0.0, 0.0, 0.0]])
        assert np.all(slice_idx_two == expected_idx_two)
        slice_idx_three = out_repr.get_param_slice(3).toarray()[:, :-1]
        expected_idx_three = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1 / 4]])
        assert np.all(slice_idx_three == expected_idx_three)
        assert out_view.get_tensor_representation(0, 4) == param_var_view.get_tensor_representation(0, 4)

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