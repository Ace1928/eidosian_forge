import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
def test_oneminusoneovern(self):
    n = np.arange(1, 20)
    x = 1.0 / n
    xm1 = 1 - 1.0 / n
    pp1 = -n * x ** (n - 1)
    pp1 -= (1 - np.sign(n - 2) ** 2) * 0.5
    dataset1 = np.column_stack([n, xm1, pp1])
    FuncData(_smirnovp, dataset1, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])