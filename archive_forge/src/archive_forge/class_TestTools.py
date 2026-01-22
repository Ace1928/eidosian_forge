from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
from statsmodels.compat.python import lrange
import string
import numpy as np
from numpy.random import standard_normal
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import longley
from statsmodels.tools import tools
from statsmodels.tools.tools import pinv_extended
class TestTools:

    def test_add_constant_list(self):
        x = lrange(1, 5)
        x = tools.add_constant(x)
        y = np.asarray([[1, 1, 1, 1], [1, 2, 3, 4.0]]).T
        assert_equal(x, y)

    def test_add_constant_1d(self):
        x = np.arange(1, 5)
        x = tools.add_constant(x)
        y = np.asarray([[1, 1, 1, 1], [1, 2, 3, 4.0]]).T
        assert_equal(x, y)

    def test_add_constant_has_constant1d(self):
        x = np.ones(5)
        x = tools.add_constant(x, has_constant='skip')
        assert_equal(x, np.ones((5, 1)))
        with pytest.raises(ValueError):
            tools.add_constant(x, has_constant='raise')
        assert_equal(tools.add_constant(x, has_constant='add'), np.ones((5, 2)))

    def test_add_constant_has_constant2d(self):
        x = np.asarray([[1, 1, 1, 1], [1, 2, 3, 4.0]]).T
        y = tools.add_constant(x, has_constant='skip')
        assert_equal(x, y)
        with pytest.raises(ValueError):
            tools.add_constant(x, has_constant='raise')
        assert_equal(tools.add_constant(x, has_constant='add'), np.column_stack((np.ones(4), x)))

    def test_add_constant_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        output = tools.add_constant(s)
        expected = pd.Series([1.0, 1.0, 1.0], name='const')
        assert_series_equal(expected, output['const'])

    def test_add_constant_dataframe(self):
        df = pd.DataFrame([[1.0, 'a', 4], [2.0, 'bc', 9], [3.0, 'def', 16]])
        output = tools.add_constant(df)
        expected = pd.Series([1.0, 1.0, 1.0], name='const')
        assert_series_equal(expected, output['const'])
        dfc = df.copy()
        dfc.insert(0, 'const', np.ones(3))
        assert_frame_equal(dfc, output)

    def test_add_constant_zeros(self):
        a = np.zeros(100)
        output = tools.add_constant(a)
        assert_equal(output[:, 0], np.ones(100))
        s = pd.Series([0.0, 0.0, 0.0])
        output = tools.add_constant(s)
        expected = pd.Series([1.0, 1.0, 1.0], name='const')
        assert_series_equal(expected, output['const'])
        df = pd.DataFrame([[0.0, 'a', 4], [0.0, 'bc', 9], [0.0, 'def', 16]])
        output = tools.add_constant(df)
        dfc = df.copy()
        dfc.insert(0, 'const', np.ones(3))
        assert_frame_equal(dfc, output)
        df = pd.DataFrame([[1.0, 'a', 0], [0.0, 'bc', 0], [0.0, 'def', 0]])
        output = tools.add_constant(df)
        dfc = df.copy()
        dfc.insert(0, 'const', np.ones(3))
        assert_frame_equal(dfc, output)

    def test_recipr(self):
        X = np.array([[2, 1], [-1, 0]])
        Y = tools.recipr(X)
        assert_almost_equal(Y, np.array([[0.5, 1], [0, 0]]))

    def test_recipr0(self):
        X = np.array([[2, 1], [-4, 0]])
        Y = tools.recipr0(X)
        assert_almost_equal(Y, np.array([[0.5, 1], [-0.25, 0]]))

    def test_extendedpinv(self):
        X = standard_normal((40, 10))
        np_inv = np.linalg.pinv(X)
        np_sing_vals = np.linalg.svd(X, 0, 0)
        sm_inv, sing_vals = pinv_extended(X)
        assert_almost_equal(np_inv, sm_inv)
        assert_almost_equal(np_sing_vals, sing_vals)

    def test_extendedpinv_singular(self):
        X = standard_normal((40, 10))
        X[:, 5] = X[:, 1] + X[:, 3]
        np_inv = np.linalg.pinv(X)
        np_sing_vals = np.linalg.svd(X, 0, 0)
        sm_inv, sing_vals = pinv_extended(X)
        assert_almost_equal(np_inv, sm_inv)
        assert_almost_equal(np_sing_vals, sing_vals)

    def test_fullrank(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X = standard_normal((40, 10))
            X[:, 0] = X[:, 1] + X[:, 2]
            Y = tools.fullrank(X)
            assert_equal(Y.shape, (40, 9))
            X[:, 5] = X[:, 3] + X[:, 4]
            Y = tools.fullrank(X)
            assert_equal(Y.shape, (40, 8))
            warnings.simplefilter('ignore')