import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
def test_smallpsf(self):
    epsilon = 0.5 ** np.arange(1, 55, 3)
    x = np.array([0.8275735551899077, 1.3163786275161036, 1.6651092133663343, 1.9525136345289607, 2.2027324540033235, 2.427292943746085, 2.6327688477341593, 2.823330050922026, 3.0018183401530627, 3.170273508408889, 3.330218444630791, 3.482825815311332, 3.629021415015205, 3.769551326282596, 3.9050272690877326, 4.035958218708255, 4.162773055788489, 4.285837174326453])
    dataset = np.column_stack([epsilon, x])
    FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()
    dataset = np.column_stack([1 - epsilon, x])
    FuncData(_kolmogci, dataset, (0,), 1, rtol=_rtol).check()