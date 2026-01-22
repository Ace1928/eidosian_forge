from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
class TestBackends:

    @staticmethod
    @pytest.fixture(params=backends)
    def backend(request):
        kwargs = {'id_to_col': {1: 0, 2: 2}, 'param_to_size': {-1: 1, 3: 1}, 'param_to_col': {3: 0, -1: 1}, 'param_size_plus_one': 2, 'var_length': 4}
        backend = CanonBackend.get_backend(request.param, **kwargs)
        assert isinstance(backend, PythonCanonBackend)
        return backend

    def test_mapping(self, backend):
        func = backend.get_func('sum')
        assert isinstance(func, Callable)
        with pytest.raises(KeyError):
            backend.get_func('notafunc')

    def test_neg(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        neg(x) means we now have
         [[-x11, -x21],
          [-x12, -x22]],

         i.e.,

         x11 x21 x12 x22
        [[-1  0   0   0],
         [0  -1   0   0],
         [0   0  -1   0],
         [0   0   0  -1]]
        """
        empty_view = backend.get_empty_view()
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, empty_view)
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        neg_lin_op = linOpHelper()
        out_view = backend.neg(neg_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        assert np.all(A == -np.eye(4))
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_transpose(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        transpose(x) means we now have
         [[x11, x21],
          [x12, x22]]

        which, when using the same columns as before, now maps to

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   0   1   0],
         [0   1   0   0],
         [0   0   0   1]]

        -> It reduces to reordering the rows of A.
        """
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        transpose_lin_op = linOpHelper((2, 2))
        out_view = backend.transpose(transpose_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_upper_tri(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        upper_tri(x) means we select only x12 (the diagonal itself is not considered).

        which, when using the same columns as before, now maps to

         x11 x21 x12 x22
        [[0   0   0   1]]

        -> It reduces to selecting a subset of the rows of A.
        """
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        upper_tri_lin_op = linOpHelper(args=[linOpHelper((2, 2))])
        out_view = backend.upper_tri(upper_tri_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 4)).toarray()
        expected = np.array([[0, 0, 1, 0]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_index(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        index() returns the subset of rows corresponding to the slicing of variables.

        e.g. x[0:2,0] yields
         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0]]

         Passing a single slice only returns the corresponding row of A.
         Note: Passing a single slice does not happen when slicing e.g. x[0], which is expanded to
         the 2d case.

         -> It reduces to selecting a subset of the rows of A.
        """
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        index_2d_lin_op = linOpHelper(data=[slice(0, 2, 1), slice(0, 1, 1)], args=[variable_lin_op])
        out_view = backend.index(index_2d_lin_op, view)
        A = out_view.get_tensor_representation(0, 2)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 4)).toarray()
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        assert np.all(A == expected)
        index_1d_lin_op = linOpHelper(data=[slice(0, 1, 1)], args=[variable_lin_op])
        out_view = backend.index(index_1d_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 4)).toarray()
        expected = np.array([[1, 0, 0, 0]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_diag_mat(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        diag_mat(x) means we select only the diagonal, i.e., x11 and x22.

        which, when using the same columns as before, now maps to

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   0   0   1]]

        -> It reduces to selecting a subset of the rows of A.
        """
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        diag_mat_lin_op = linOpHelper(shape=(2, 2), data=0)
        out_view = backend.diag_mat(diag_mat_lin_op, view)
        A = out_view.get_tensor_representation(0, 2)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 4)).toarray()
        expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_diag_mat_with_offset(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        diag_mat(x, k=1) means we select only the 1-(super)diagonal, i.e., x12.

        which, when using the same columns as before, now maps to

         x11 x21 x12 x22
        [[0   0   1   0]]

        -> It reduces to selecting a subset of the rows of A.
        """
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        k = 1
        diag_mat_lin_op = linOpHelper(shape=(1, 1), data=k)
        out_view = backend.diag_mat(diag_mat_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 4)).toarray()
        expected = np.array([[0, 0, 1, 0]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_diag_vec(self, backend):
        """
        define x = Variable((2,)) with
        [x1, x2]

        x is represented as eye(2) in the A matrix, i.e.,

         x1  x2
        [[1  0],
         [0  1]]

        diag_vec(x) means we introduce zero rows as if the vector was the diagonal
        of an n x n matrix, with n the length of x.

        Thus, when using the same columns as before, we now have

         x1  x2
        [[1  0],
         [0  0],
         [0  0],
         [0  1]]
        """
        variable_lin_op = linOpHelper((2,), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 2)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))
        diag_vec_lin_op = linOpHelper(shape=(2, 2), data=0)
        out_view = backend.diag_vec(diag_vec_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 2)).toarray()
        expected = np.array([[1, 0], [0, 0], [0, 0], [0, 1]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_diag_vec_with_offset(self, backend):
        """
        define x = Variable((2,)) with
        [x1, x2]

        x is represented as eye(2) in the A matrix, i.e.,

         x1  x2
        [[1  0],
         [0  1]]

        diag_vec(x, k) means we introduce zero rows as if the vector was the k-diagonal
        of an n+|k| x n+|k| matrix, with n the length of x.

        Thus, for k=1 and using the same columns as before, want to represent
        [[0  x1 0],
        [ 0  0  x2],
        [[0  0  0]]
        i.e., unrolled in column-major order:

         x1  x2
        [[0  0],
        [0  0],
        [0  0],
        [1  0],
        [0  0],
        [0  0],
        [0  0],
        [0  1],
        [0  0]]
        """
        variable_lin_op = linOpHelper((2,), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 2)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))
        k = 1
        diag_vec_lin_op = linOpHelper(shape=(3, 3), data=k)
        out_view = backend.diag_vec(diag_vec_lin_op, view)
        A = out_view.get_tensor_representation(0, 9)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(9, 2)).toarray()
        expected = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 9) == view.get_tensor_representation(0, 9)

    def test_sum_entries(self, backend):
        """
        define x = Variable((2,)) with
        [x1, x2]

        x is represented as eye(2) in the A matrix, i.e.,

         x1  x2
        [[1  0],
         [0  1]]

        sum_entries(x) means we consider the entries in all rows, i.e., we sum along the row axis.

        Thus, when using the same columns as before, we now have

         x1  x2
        [[1  1]]
        """
        variable_lin_op = linOpHelper((2,), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 2)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))
        sum_entries_lin_op = linOpHelper()
        out_view = backend.sum_entries(sum_entries_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 2)).toarray()
        expected = np.array([[1, 1]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_promote(self, backend):
        """
        define x = Variable((1,)) with
        [x1,]

        x is represented as eye(1) in the A matrix, i.e.,

         x1
        [[1]]

        promote(x) means we repeat the row to match the required dimensionality of n rows.

        Thus, when using the same columns as before and assuming n = 3, we now have

         x1
        [[1],
         [1],
         [1]]
        """
        variable_lin_op = linOpHelper((1,), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 1)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(1, 1)).toarray()
        assert np.all(view_A == np.eye(1))
        promote_lin_op = linOpHelper((3,))
        out_view = backend.promote(promote_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(3, 1)).toarray()
        expected = np.array([[1], [1], [1]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_hstack(self, backend):
        """
        define x,y = Variable((1,)), Variable((1,))

        hstack([x, y]) means the expression should be represented in the A matrix as if it
        was a Variable of shape (2,), i.e.,

          x  y
        [[1  0],
         [0  1]]
        """
        lin_op_x = linOpHelper((1,), type='variable', data=1)
        lin_op_y = linOpHelper((1,), type='variable', data=2)
        hstack_lin_op = linOpHelper(args=[lin_op_x, lin_op_y])
        backend.id_to_col = {1: 0, 2: 1}
        out_view = backend.hstack(hstack_lin_op, backend.get_empty_view())
        A = out_view.get_tensor_representation(0, 2)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 2)).toarray()
        expected = np.eye(2)
        assert np.all(A == expected)

    def test_vstack(self, backend):
        """
        define x,y = Variable((1,2)), Variable((1,2)) with
        [[x1, x2]]
        and
        [[y1, y2]]

        vstack([x, y]) yields

        [[x1, x2],
         [y1, y2]]

        which maps to

         x1   x2  y1  y2
        [[1   0   0   0],
         [0   0   1   0],
         [0   1   0   0],
         [0   0   0   1]]
        """
        lin_op_x = linOpHelper((1, 2), type='variable', data=1)
        lin_op_y = linOpHelper((1, 2), type='variable', data=2)
        vstack_lin_op = linOpHelper(args=[lin_op_x, lin_op_y])
        backend.id_to_col = {1: 0, 2: 2}
        out_view = backend.vstack(vstack_lin_op, backend.get_empty_view())
        A = out_view.get_tensor_representation(0, 4)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        assert np.all(A == expected)

    def test_mul(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

         Multiplying with the constant from the left
        [[1, 2],
         [3, 4]],

         we expect the output to be
        [[  x11 + 2 x21,   x12 + 2 x22],
         [3 x11 + 4 x21, 3 x12 + 4 x22]]

        i.e., when represented in the A matrix (again using column-major order):
         x11 x21 x12 x22
        [[1   2   0   0],
         [3   4   0   0],
         [0   0   1   2],
         [0   0   3   4]]
        """
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        lhs = linOpHelper((2, 2), type='dense_const', data=np.array([[1, 2], [3, 4]]))
        mul_lin_op = linOpHelper(data=lhs, args=[variable_lin_op])
        out_view = backend.mul(mul_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 2], [0, 0, 3, 4]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_rmul(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

         Multiplying with the constant from the right
         (intentionally using 1D vector to cover edge case)
        [1, 2]

         we expect the output to be
         [[x11 + 2 x12],
          [x21 + 2 x22]]

        i.e., when represented in the A matrix:
         x11 x21 x12 x22
        [[1   0   2   0],
         [0   1   0   2]]
        """
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        rhs = linOpHelper((2,), type='dense_const', data=np.array([1, 2]))
        rmul_lin_op = linOpHelper(data=rhs, args=[variable_lin_op])
        out_view = backend.rmul(rmul_lin_op, view)
        A = out_view.get_tensor_representation(0, 2)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 4)).toarray()
        expected = np.array([[1, 0, 2, 0], [0, 1, 0, 2]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 2) == view.get_tensor_representation(0, 2)

    def test_mul_elementwise(self, backend):
        """
        define x = Variable((2,)) with
        [x1, x2]

        x is represented as eye(2) in the A matrix, i.e.,

         x1  x2
        [[1  0],
         [0  1]]

         mul_elementwise(x, a) means 'a' is reshaped into a column vector and multiplied by A.
         E.g. for a = (2,3), we obtain

         x1  x2
        [[2  0],
         [0  3]]
        """
        variable_lin_op = linOpHelper((2,), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 2)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))
        lhs = linOpHelper((2,), type='dense_const', data=np.array([2, 3]))
        mul_elementwise_lin_op = linOpHelper(data=lhs)
        out_view = backend.mul_elem(mul_elementwise_lin_op, view)
        A = out_view.get_tensor_representation(0, 2)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 2)).toarray()
        expected = np.array([[2, 0], [0, 3]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 2) == view.get_tensor_representation(0, 2)

    def test_div(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

         Dividing elementwise with
        [[1, 2],
         [3, 4]],

        we obtain:
         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1/3 0   0],
         [0   0   1/2 0],
         [0   0   0   1/4]]
        """
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        lhs = linOpHelper((2, 2), type='dense_const', data=np.array([[1, 2], [3, 4]]))
        div_lin_op = linOpHelper(data=lhs)
        out_view = backend.div(div_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array([[1, 0, 0, 0], [0, 1 / 3, 0, 0], [0, 0, 1 / 2, 0], [0, 0, 0, 1 / 4]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_trace(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        trace(x) means we sum the diagonal entries of x, i.e.

         x11 x21 x12 x22
        [[1   0   0   1]]
        """
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        trace_lin_op = linOpHelper(args=[variable_lin_op])
        out_view = backend.trace(trace_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 4)).toarray()
        expected = np.array([[1, 0, 0, 1]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_conv(self, backend):
        """
        define x = Variable((3,)) with
        [x1, x2, x3]

        having f = [1,2,3], conv(f, x) means we repeat the column vector of f for each column in
        the A matrix, shifting it down by one after each repetition, i.e.,
          x1 x2 x3
        [[1  0  0],
         [2  1  0],
         [3  2  1],
         [0  3  2],
         [0  0  3]]
        """
        variable_lin_op = linOpHelper((3,), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 3)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(3, 3)).toarray()
        assert np.all(view_A == np.eye(3))
        f = linOpHelper((3,), type='dense_const', data=np.array([1, 2, 3]))
        conv_lin_op = linOpHelper(data=f, shape=(5, 1), args=[variable_lin_op])
        out_view = backend.conv(conv_lin_op, view)
        A = out_view.get_tensor_representation(0, 5)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(5, 3)).toarray()
        expected = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0], [0.0, 3.0, 2.0], [0.0, 0.0, 3.0]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 5) == view.get_tensor_representation(0, 5)

    def test_kron_r(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        and
        a = [[1],
             [2]],

        kron(a, x) means we have
        [[x11, x12],
         [x21, x22],
         [2x11, 2x12],
         [2x21, 2x22]]

        i.e. as represented in the A matrix (again in column-major order)

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [2   0   0   0],
         [0   2   0   0],
         [0   0   1   0],
         [0   0   0   1],
         [0   0   2   0],
         [0   0   0   2]]

        However computing kron(a, x) (where x is represented as eye(4))
        directly gives us:
        [[1   0   0   0],
         [2   0   0   0],
         [0   1   0   0],
         [0   2   0   0],
         [0   0   1   0],
         [0   0   2   0],
         [0   0   0   1],
         [0   0   0   2]]
        So we must swap the row indices of the resulting matrix.
        """
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        a = linOpHelper((2, 1), type='dense_const', data=np.array([[1], [2]]))
        kron_r_lin_op = linOpHelper(data=a, args=[variable_lin_op])
        out_view = backend.kron_r(kron_r_lin_op, view)
        A = out_view.get_tensor_representation(0, 8)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(8, 4)).toarray()
        expected = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 8) == view.get_tensor_representation(0, 8)

    def test_kron_l(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        and
        a = [[1, 2]],

        kron(x, a) means we have
        [[x11, 2x11, x12, 2x12],
         [x21, 2x21, x22, 2x22]]

        i.e. as represented in the A matrix (again in column-major order)

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [2   0   0   0],
         [0   2   0   0],
         [0   0   1   0],
         [0   0   0   1],
         [0   0   2   0],
         [0   0   0   2]]

         However computing kron(x, a) (where a is reshaped into a column vector
         and x is represented as eye(4)) directly gives us:
        [[1   0   0   0],
         [2   0   0   0],
         [0   1   0   0],
         [0   2   0   0],
         [0   0   1   0],
         [0   0   2   0],
         [0   0   0   1],
         [0   0   0   2]]
        So we must swap the row indices of the resulting matrix.
        """
        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))
        a = linOpHelper((1, 2), type='dense_const', data=np.array([[1, 2]]))
        kron_l_lin_op = linOpHelper(data=a, args=[variable_lin_op])
        out_view = backend.kron_l(kron_l_lin_op, view)
        A = out_view.get_tensor_representation(0, 8)
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(8, 4)).toarray()
        expected = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
        assert np.all(A == expected)
        assert out_view.get_tensor_representation(0, 8) == view.get_tensor_representation(0, 8)

    def test_get_kron_row_indices(self, backend):
        """
        kron(l,r)
        with
        l = [[x1, x3],  r = [[a],
             [x2, x4]]       [b]]

        yields
        [[ax1, ax3],
         [bx1, bx3],
         [ax2, ax4],
         [bx2, bx4]]

        Which is what we get when we compute kron(l,r) directly,
        as l is represented as eye(4) and r is reshaped into a column vector.

        So we have:
        kron(l,r) =
        [[a, 0, 0, 0],
         [b, 0, 0, 0],
         [0, a, 0, 0],
         [0, b, 0, 0],
         [0, 0, a, 0],
         [0, 0, b, 0],
         [0, 0, 0, a],
         [0, 0, 0, b]].

        Thus, this function should return arange(8).
        """
        indices = backend._get_kron_row_indices((2, 2), (2, 1))
        assert np.all(indices == np.arange(8))
        '\n        kron(l,r)\n        with \n        l = [[x1],  r = [[a, c],\n             [x2]]       [b, d]]\n\n        yields\n        [[ax1, cx1],\n         [bx1, dx1],\n         [ax2, cx2],\n         [bx2, dx2]]\n        \n        Here, we have to swap the row indices of the resulting matrix.\n        Immediately applying kron(l,r) gives to eye(2) and r reshaped to \n        a column vector gives.\n                 \n        So we have:\n        kron(l,r) = \n        [[a, 0],\n         [b, 0],\n         [c, 0],\n         [d, 0],\n         [0, a],\n         [0, b]\n         [0, c],\n         [0, d]].\n\n        Thus, we need to to return [0, 1, 4, 5, 2, 3, 6, 7].\n        '
        indices = backend._get_kron_row_indices((2, 1), (2, 2))
        assert np.all(indices == [0, 1, 4, 5, 2, 3, 6, 7])
        indices = backend._get_kron_row_indices((1, 2), (3, 2))
        assert np.all(indices == np.arange(12))
        indices = backend._get_kron_row_indices((3, 2), (1, 2))
        assert np.all(indices == [0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11])
        indices = backend._get_kron_row_indices((2, 2), (2, 2))
        expected = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        assert np.all(indices == expected)

    def test_tensor_view_combine_potentially_none(self, backend):
        view = backend.get_empty_view()
        assert view.combine_potentially_none(None, None) is None
        a = {'a': [1]}
        b = {'b': [2]}
        assert view.combine_potentially_none(a, None) == a
        assert view.combine_potentially_none(None, a) == a
        assert view.combine_potentially_none(a, b) == view.add_dicts(a, b)