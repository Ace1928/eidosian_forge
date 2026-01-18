import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
def test_smallpcdf(self):
    epsilon = 0.5 ** np.arange(1, 55, 3)
    x = np.array([0.8275735551899077, 0.5345255069097583, 0.4320114038786941, 0.3736868442620478, 0.3345161714909591, 0.3057833329315859, 0.2835052890528936, 0.2655578150208676, 0.2506869966107999, 0.2380971058736669, 0.2272549289962079, 0.217787636160004, 0.2094254686862041, 0.2019676748836232, 0.1952612948137504, 0.1891874239646641, 0.1836520225050326, 0.1785795904846466])
    dataset = np.column_stack([1 - epsilon, x])
    FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()
    dataset = np.column_stack([epsilon, x])
    FuncData(_kolmogci, dataset, (0,), 1, rtol=_rtol).check()