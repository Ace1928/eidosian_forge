import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_shuffle_dont_convert_to_array(csc_container):
    a = ['a', 'b', 'c']
    b = np.array(['a', 'b', 'c'], dtype=object)
    c = [1, 2, 3]
    d = MockDataFrame(np.array([['a', 0], ['b', 1], ['c', 2]], dtype=object))
    e = csc_container(np.arange(6).reshape(3, 2))
    a_s, b_s, c_s, d_s, e_s = shuffle(a, b, c, d, e, random_state=0)
    assert a_s == ['c', 'b', 'a']
    assert type(a_s) == list
    assert_array_equal(b_s, ['c', 'b', 'a'])
    assert b_s.dtype == object
    assert c_s == [3, 2, 1]
    assert type(c_s) == list
    assert_array_equal(d_s, np.array([['c', 2], ['b', 1], ['a', 0]], dtype=object))
    assert type(d_s) == MockDataFrame
    assert_array_equal(e_s.toarray(), np.array([[4, 5], [2, 3], [0, 1]]))