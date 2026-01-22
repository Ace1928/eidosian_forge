import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_
import pytest
import statsmodels.stats.weightstats as smws
from statsmodels.tools.testing import Holder
class TestTosti1(CheckTostMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = tost_clinic_indep_1
        x, y = (clinic[:15, 2], clinic[15:, 2])
        cls.res1 = Holder()
        res = smws.ttost_ind(x, y, -0.6, 0.6, usevar='unequal')
        cls.res1.pvalue = res[0]