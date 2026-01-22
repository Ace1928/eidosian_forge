from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
class TestSciPyBackend:

    @staticmethod
    @pytest.fixture()
    def scipy_backend():
        kwargs = {'id_to_col': {1: 0}, 'param_to_size': {-1: 1, 2: 2}, 'param_to_col': {2: 0, -1: 2}, 'param_size_plus_one': 3, 'var_length': 2}
        backend = CanonBackend.get_backend(s.SCIPY_CANON_BACKEND, **kwargs)
        assert isinstance(backend, SciPyCanonBackend)
        return backend

    def test_get_variable_tensor(self, scipy_backend):
        outer = scipy_backend.get_variable_tensor((2,), 1)
        assert outer.keys() == {1}, 'Should only be in variable with ID 1'
        inner = outer[1]
        assert inner.keys() == {-1}, 'Should only be in parameter slice -1, i.e. non parametrized.'
        tensor = inner[-1]
        assert isinstance(tensor, sp.spmatrix), 'Should be a scipy sparse matrix'
        assert tensor.shape == (2, 2), 'Should be a 1*2x2 tensor'
        assert np.all(tensor == np.eye(2)), 'Should be eye(2)'

    @pytest.mark.parametrize('data', [np.array([[1, 2], [3, 4]]), sp.eye(2) * 4])
    def test_get_data_tensor(self, scipy_backend, data):
        outer = scipy_backend.get_data_tensor(data)
        assert outer.keys() == {-1}, 'Should only be constant variable ID.'
        inner = outer[-1]
        assert inner.keys() == {-1}, 'Should only be in parameter slice -1, i.e. non parametrized.'
        tensor = inner[-1]
        assert isinstance(tensor, sp.spmatrix), 'Should be a scipy sparse matrix'
        assert tensor.shape == (4, 1), 'Should be a 1*4x1 tensor'
        expected = sp.csr_matrix(data.reshape((-1, 1), order='F'))
        assert (tensor != expected).nnz == 0

    def test_get_param_tensor(self, scipy_backend):
        shape = (2, 2)
        size = np.prod(shape)
        scipy_backend.param_to_size = {-1: 1, 3: 4}
        outer = scipy_backend.get_param_tensor(shape, 3)
        assert outer.keys() == {-1}, 'Should only be constant variable ID.'
        inner = outer[-1]
        assert inner.keys() == {3}, 'Should only be the parameter slice of parameter with id 3.'
        tensor = inner[3]
        assert isinstance(tensor, sp.spmatrix), 'Should be a scipy sparse matrix'
        assert tensor.shape == (16, 1), 'Should be a 4*4x1 tensor'
        assert (tensor.reshape((size, size)) != sp.eye(size, format='csr')).nnz == 0, 'Should be eye(4) when reshaping'

    def test_tensor_view_add_dicts(self, scipy_backend):
        view = scipy_backend.get_empty_view()
        one = sp.eye(1)
        two = sp.eye(1) * 2
        three = sp.eye(1) * 3
        assert view.add_dicts({}, {}) == {}
        assert view.add_dicts({'a': one}, {'a': two}) == {'a': three}
        assert view.add_dicts({'a': one}, {'b': two}) == {'a': one, 'b': two}
        assert view.add_dicts({'a': {'c': one}}, {'a': {'c': one}}) == {'a': {'c': two}}
        with pytest.raises(ValueError, match="Values must either be dicts or <class 'scipy.sparse."):
            view.add_dicts({'a': 1}, {'a': 2})

    @staticmethod
    @pytest.mark.parametrize('shape', [(1, 1), (2, 2), (3, 3), (4, 4)])
    def test_stacked_kron_r(shape, scipy_backend):
        p = 2
        reps = 3
        param_id = 2
        matrices = [sp.random(*shape, random_state=i, density=0.5) for i in range(p)]
        stacked = sp.vstack(matrices)
        repeated = scipy_backend._stacked_kron_r({param_id: stacked}, reps)
        repeated = repeated[param_id]
        expected = sp.vstack([sp.kron(sp.eye(reps), m) for m in matrices])
        assert (expected != repeated).nnz == 0

    @staticmethod
    @pytest.mark.parametrize('shape', [(1, 1), (2, 2), (3, 3), (4, 4)])
    def test_stacked_kron_l(shape, scipy_backend):
        p = 2
        reps = 3
        param_id = 2
        matrices = [sp.random(*shape, random_state=i, density=0.5) for i in range(p)]
        stacked = sp.vstack(matrices)
        repeated = scipy_backend._stacked_kron_l({param_id: stacked}, reps)
        repeated = repeated[param_id]
        expected = sp.vstack([sp.kron(m, sp.eye(reps)) for m in matrices])
        assert (expected != repeated).nnz == 0

    @staticmethod
    def test_reshape_single_constant_tensor(scipy_backend):
        a = sp.csc_matrix(np.tile(np.arange(6), 3).reshape((-1, 1)))
        reshaped = scipy_backend._reshape_single_constant_tensor(a, (3, 2))
        expected = np.arange(6).reshape((3, 2), order='F')
        expected = sp.csc_matrix(np.tile(expected, (3, 1)))
        assert (reshaped != expected).nnz == 0

    @staticmethod
    @pytest.mark.parametrize('shape', [(1, 1), (2, 2), (3, 2), (2, 3)])
    def test_transpose_stacked(shape, scipy_backend):
        p = 2
        param_id = 2
        matrices = [sp.random(*shape, random_state=i, density=0.5) for i in range(p)]
        stacked = sp.vstack(matrices)
        transposed = scipy_backend._transpose_stacked(stacked, param_id)
        expected = sp.vstack([m.T for m in matrices])
        assert (expected != transposed).nnz == 0