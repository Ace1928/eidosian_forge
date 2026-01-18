import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
def test_x_equals_0(self):
    dataset = [(n, 0, 1) for n in itertools.chain(range(2, 20), range(1010, 1020))]
    dataset = np.asarray(dataset)
    FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
    dataset[:, 1] = 1 - dataset[:, 1]
    FuncData(_smirnovci, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])