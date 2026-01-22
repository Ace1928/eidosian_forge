import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.sandbox.distributions.extras import (
class TestAbsNormHalfNorm(CheckDistEquivalence):

    def __init__(self):
        self.dist = absnormalg
        self.trargs = ()
        self.trkwds = {}
        self.statsdist = stats.halfnorm
        self.stargs = ()
        self.stkwds = {}