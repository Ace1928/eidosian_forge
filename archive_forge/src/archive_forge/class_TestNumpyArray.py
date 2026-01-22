import random
from copy import copy
import pytest
import networkx as nx
from networkx.utils import (
from networkx.utils.misc import _dict_to_numpy_array1, _dict_to_numpy_array2
class TestNumpyArray:

    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip('numpy')

    def test_numpy_to_list_of_ints(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([1.0, 2, 3])
        c = np.array([1.1, 2, 3])
        assert type(make_list_of_ints(a)) == list
        assert make_list_of_ints(b) == list(b)
        B = make_list_of_ints(b)
        assert type(B[0]) == int
        pytest.raises(nx.NetworkXError, make_list_of_ints, c)

    def test__dict_to_numpy_array1(self):
        d = {'a': 1, 'b': 2}
        a = _dict_to_numpy_array1(d, mapping={'a': 0, 'b': 1})
        np.testing.assert_allclose(a, np.array([1, 2]))
        a = _dict_to_numpy_array1(d, mapping={'b': 0, 'a': 1})
        np.testing.assert_allclose(a, np.array([2, 1]))
        a = _dict_to_numpy_array1(d)
        np.testing.assert_allclose(a.sum(), 3)

    def test__dict_to_numpy_array2(self):
        d = {'a': {'a': 1, 'b': 2}, 'b': {'a': 10, 'b': 20}}
        mapping = {'a': 1, 'b': 0}
        a = _dict_to_numpy_array2(d, mapping=mapping)
        np.testing.assert_allclose(a, np.array([[20, 10], [2, 1]]))
        a = _dict_to_numpy_array2(d)
        np.testing.assert_allclose(a.sum(), 33)

    def test_dict_to_numpy_array_a(self):
        d = {'a': {'a': 1, 'b': 2}, 'b': {'a': 10, 'b': 20}}
        mapping = {'a': 0, 'b': 1}
        a = dict_to_numpy_array(d, mapping=mapping)
        np.testing.assert_allclose(a, np.array([[1, 2], [10, 20]]))
        mapping = {'a': 1, 'b': 0}
        a = dict_to_numpy_array(d, mapping=mapping)
        np.testing.assert_allclose(a, np.array([[20, 10], [2, 1]]))
        a = _dict_to_numpy_array2(d)
        np.testing.assert_allclose(a.sum(), 33)

    def test_dict_to_numpy_array_b(self):
        d = {'a': 1, 'b': 2}
        mapping = {'a': 0, 'b': 1}
        a = dict_to_numpy_array(d, mapping=mapping)
        np.testing.assert_allclose(a, np.array([1, 2]))
        a = _dict_to_numpy_array1(d)
        np.testing.assert_allclose(a.sum(), 3)