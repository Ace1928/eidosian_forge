from statsmodels.compat.python import asbytes
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
class TestTuckeyHSD3(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(cls):
        cls.endog = dta3['Relief']
        cls.groups = dta3['Brand']
        cls.alpha = 0.05
        cls.setup_class_()
        cls.meandiff2 = sas_['mean']
        cls.confint2 = sas_[['lower', 'upper']].astype(float).values.reshape((3, 2))
        cls.reject2 = sas_['sig'] == asbytes('***')