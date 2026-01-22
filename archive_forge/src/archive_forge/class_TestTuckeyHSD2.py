from statsmodels.compat.python import asbytes
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
class TestTuckeyHSD2(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(cls):
        cls.endog = dta2['StressReduction']
        cls.groups = dta2['Treatment']
        cls.alpha = 0.05
        cls.setup_class_()
        tukeyhsd2s = np.array([1.5, 1, -0.5, 0.3214915, -0.1785085, -1.678509, 2.678509, 2.178509, 0.6785085, 0.01056279, 0.1079035, 0.5513904]).reshape(3, 4, order='F')
        cls.meandiff2 = tukeyhsd2s[:, 0]
        cls.confint2 = tukeyhsd2s[:, 1:3]
        pvals = tukeyhsd2s[:, 3]
        cls.reject2 = pvals < 0.05

    def test_table_names_default_group_order(self):
        t = self.res._results_table
        expected_order = [(b'medical', b'mental'), (b'medical', b'physical'), (b'mental', b'physical')]
        for i in range(1, 4):
            first_group = t[i][0].data
            second_group = t[i][1].data
            assert_((first_group, second_group) == expected_order[i - 1])

    def test_table_names_custom_group_order(self):
        mc = MultiComparison(self.endog, self.groups, group_order=[b'physical', b'medical', b'mental'])
        res = mc.tukeyhsd(alpha=self.alpha)
        t = res._results_table
        expected_order = [(b'physical', b'medical'), (b'physical', b'mental'), (b'medical', b'mental')]
        for i in range(1, 4):
            first_group = t[i][0].data
            second_group = t[i][1].data
            assert_((first_group, second_group) == expected_order[i - 1])