import os
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.regime_switching import (markov_switching,
class TestMumpspcNoconstL1Variance(MarkovRegression):

    @classmethod
    def setup_class(cls):
        true = {'params': np.r_[0.762733, 0.1473767, 0.420275, 0.9847369, 0.0562405 ** 2, 0.2611362 ** 2], 'llf': 131.7225, 'llf_fit': 131.7225, 'llf_fit_em': 131.7175}
        super().setup_class(true, mumpspc[1:], k_regimes=2, trend='n', exog=mumpspc[:-1], switching_variance=True, atol=0.0001)