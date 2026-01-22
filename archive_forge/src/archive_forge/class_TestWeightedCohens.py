import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from statsmodels.stats.inter_rater import (fleiss_kappa, cohens_kappa,
from statsmodels.tools.testing import Holder
class TestWeightedCohens(CheckCohens):

    @classmethod
    def setup_class(cls):
        cls.res = cohens_kappa(table10, weights=[0, 1, 2])
        res10w_sas = [0.4701, 0.1457, 0.1845, 0.7558]
        res10w_sash0 = [0.1426, 3.2971, 0.0005, 0.001]
        cls.res2 = res10w_sas + res10w_sash0
        cls.res_string = '                  Weighted Kappa Coefficient\n              --------------------------------\n              Kappa                     0.4701\n              ASE                       0.1457\n              95% Lower Conf Limit      0.1845\n              95% Upper Conf Limit      0.7558\n\n                 Test of H0: Weighted Kappa = 0\n\n              ASE under H0              0.1426\n              Z                         3.2971\n              One-sided Pr >  Z         0.0005\n              Two-sided Pr > |Z|        0.0010' + '\n'

    def test_option(self):
        kappa = cohens_kappa(table10, weights=[0, 1, 2], return_results=False)
        assert_almost_equal(kappa, self.res2[0], decimal=4)