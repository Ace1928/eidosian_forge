import os
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.regime_switching import (markov_switching,
class TestFedFundsConstL1Exog3(MarkovRegression):

    @classmethod
    def setup_class(cls):
        true = {'params': np.r_[0.7253684, 0.1641252, 0.6178282, 0.2564055, 0.7994204, 0.3821718, 0.5261292, -0.0034106, 0.6015991, 0.8464551, 0.9690088, 0.4178913, 0.1201952, 0.0464136, 0.1075357, -0.0425603, 0.1298906, 0.9099168, 0.438375 ** 2], 'llf': -189.89493, 'llf_fit': -182.27188, 'llf_fit_em': -226.88581}
        super().setup_class(true, fedfunds[4:], k_regimes=3, exog=np.c_[fedfunds[3:-1], ogap[4:], inf[4:]])

    def test_fit(self, **kwargs):
        kwargs['search_reps'] = 20
        np.random.seed(1234)
        super().test_fit(**kwargs)