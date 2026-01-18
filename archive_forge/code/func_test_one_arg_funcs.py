import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_one_arg_funcs(self):
    funcs = (random.exponential, random.standard_gamma, random.chisquare, random.standard_t, random.pareto, random.weibull, random.power, random.rayleigh, random.poisson, random.zipf, random.geometric, random.logseries)
    probfuncs = (random.geometric, random.logseries)
    for func in funcs:
        if func in probfuncs:
            out = func(np.array([0.5]))
        else:
            out = func(self.argOne)
        assert_equal(out.shape, self.tgtShape)