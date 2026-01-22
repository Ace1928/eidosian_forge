import os
import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tsa.vector_ar.vecm import coint_johansen
class TestCointJoh25(CheckCointJoh):

    @classmethod
    def setup_class(cls):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=HypothesisTestWarning)
            cls.res = coint_johansen(dta, 2, 5)
        cls.nobs_r = 173 - 1 - 5
        cls.res1_m = np.array([270.1887263915158, 171.6870096307863, 107.8613367358704, 70.82424032233558, 44.62551818267534, 25.74352073857572, 14.17882426926978, 4.288656185006764, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cls.res1_m[cls.res1_m == 0] = np.nan
        cls.res2_m = np.array([98.50171676072955, 63.82567289491584, 37.03709641353485, 26.19872213966024, 18.88199744409963, 11.56469646930594, 9.890168084263012, 4.288656185006764, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cls.res2_m[cls.res2_m == 0] = np.nan