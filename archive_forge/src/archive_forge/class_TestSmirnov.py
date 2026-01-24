import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
class TestSmirnov:

    def test_nan(self):
        assert_(np.isnan(smirnov(1, np.nan)))

    def test_basic(self):
        dataset = [(1, 0.1, 0.9), (1, 0.875, 0.125), (2, 0.875, 0.125 * 0.125), (3, 0.875, 0.125 * 0.125 * 0.125)]
        dataset = np.asarray(dataset)
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_x_equals_0(self):
        dataset = [(n, 0, 1) for n in itertools.chain(range(2, 20), range(1010, 1020))]
        dataset = np.asarray(dataset)
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_x_equals_1(self):
        dataset = [(n, 1, 0) for n in itertools.chain(range(2, 20), range(1010, 1020))]
        dataset = np.asarray(dataset)
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_x_equals_0point5(self):
        dataset = [(1, 0.5, 0.5), (2, 0.5, 0.25), (3, 0.5, 0.166666666667), (4, 0.5, 0.09375), (5, 0.5, 0.056), (6, 0.5, 0.0327932098928), (7, 0.5, 0.0191958707681), (8, 0.5, 0.0112953186035), (9, 0.5, 0.00661933257355), (10, 0.5, 0.003888705)]
        dataset = np.asarray(dataset)
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_n_equals_1(self):
        x = np.linspace(0, 1, 101, endpoint=True)
        dataset = np.column_stack([[1] * len(x), x, 1 - x])
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_n_equals_2(self):
        x = np.linspace(0.5, 1, 101, endpoint=True)
        p = np.power(1 - x, 2)
        n = np.array([2] * len(x))
        dataset = np.column_stack([n, x, p])
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_n_equals_3(self):
        x = np.linspace(0.7, 1, 31, endpoint=True)
        p = np.power(1 - x, 3)
        n = np.array([3] * len(x))
        dataset = np.column_stack([n, x, p])
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_n_large(self):
        x = 0.4
        pvals = np.array([smirnov(n, x) for n in range(400, 1100, 20)])
        dfs = np.diff(pvals)
        assert_(np.all(dfs <= 0), msg='Not all diffs negative %s' % dfs)