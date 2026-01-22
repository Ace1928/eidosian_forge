from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
class TestNumPyBackend:

    @staticmethod
    @pytest.fixture()
    def numpy_backend():
        kwargs = {'id_to_col': {1: 0}, 'param_to_size': {-1: 1, 2: 2}, 'param_to_col': {2: 0, -1: 2}, 'param_size_plus_one': 3, 'var_length': 2}
        backend = CanonBackend.get_backend(s.NUMPY_CANON_BACKEND, **kwargs)
        assert isinstance(backend, NumPyCanonBackend)
        return backend

    def test_get_variable_tensor(self, numpy_backend):
        outer = numpy_backend.get_variable_tensor((2,), 1)
        assert outer.keys() == {1}, 'Should only be in variable with ID 1'
        inner = outer[1]
        assert inner.keys() == {-1}, 'Should only be in parameter slice -1, i.e. non parametrized.'
        tensor = inner[-1]
        assert isinstance(tensor, np.ndarray), 'Should be a numpy array'
        assert tensor.shape == (1, 2, 2), 'Should be a 1x2x2 tensor'
        assert np.all(tensor[0] == np.eye(2)), 'Should be eye(2)'

    @pytest.mark.parametrize('data', [np.array([[1, 2], [3, 4]]), sp.eye(2) * 4])
    def test_get_data_tensor(self, numpy_backend, data):
        outer = numpy_backend.get_data_tensor(data)
        assert outer.keys() == {-1}, 'Should only be constant variable ID.'
        inner = outer[-1]
        assert inner.keys() == {-1}, 'Should only be in parameter slice -1, i.e. non parametrized.'
        tensor = inner[-1]
        assert isinstance(tensor, np.ndarray), 'Should be a numpy array'
        assert isinstance(tensor[0], np.ndarray), 'Inner matrix should also be a numpy array'
        assert tensor.shape == (1, 4, 1), 'Should be a 1x4x1 tensor'
        expected = numpy_backend._to_dense(data).reshape((-1, 1), order='F')
        assert np.all(tensor[0] == expected)

    def test_get_param_tensor(self, numpy_backend):
        shape = (2, 2)
        size = np.prod(shape)
        outer = numpy_backend.get_param_tensor(shape, 3)
        assert outer.keys() == {-1}, 'Should only be constant variable ID.'
        inner = outer[-1]
        assert inner.keys() == {3}, 'Should only be the parameter slice of parameter with id 3.'
        tensor = inner[3]
        assert isinstance(tensor, np.ndarray), 'Should be a numpy array'
        assert tensor.shape == (4, 4, 1), 'Should be a 4x4x1 tensor'
        assert np.all(tensor[:, :, 0] == np.eye(size)), 'Should be eye(4) along axes 1 and 2'

    def test_tensor_view_add_dicts(self, numpy_backend):
        view = numpy_backend.get_empty_view()
        one = np.array([1])
        two = np.array([2])
        three = np.array([3])
        assert view.add_dicts({}, {}) == {}
        assert view.add_dicts({'a': one}, {'a': two}) == {'a': three}
        assert view.add_dicts({'a': one}, {'b': two}) == {'a': one, 'b': two}
        assert view.add_dicts({'a': {'c': one}}, {'a': {'c': one}}) == {'a': {'c': two}}
        with pytest.raises(ValueError, match="Values must either be dicts or <class 'numpy.ndarray'>"):
            view.add_dicts({'a': 1}, {'a': 2})