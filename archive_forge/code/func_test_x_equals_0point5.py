import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
def test_x_equals_0point5(self):
    dataset = [(1, 0.5, 0.5), (2, 0.5, 0.366025403784), (2, 0.25, 0.5), (3, 0.5, 0.297156508177), (4, 0.5, 0.255520481121), (5, 0.5, 0.234559536069), (6, 0.5, 0.21715965898), (7, 0.5, 0.202722580034), (8, 0.5, 0.190621765256), (9, 0.5, 0.180363501362), (10, 0.5, 0.17157867006)]
    dataset = np.asarray(dataset)
    FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
    dataset[:, 1] = 1 - dataset[:, 1]
    FuncData(_smirnovci, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])