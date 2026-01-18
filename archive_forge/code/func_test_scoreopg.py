import numpy as np
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.stats._diagnostic_other import CMTNewey, CMTTauchen
import statsmodels.stats._diagnostic_other as diao
def test_scoreopg(self):
    expected = self.results_opg
    for msg, actual in self.res_opg():
        assert_allclose(actual, expected[:np.size(actual)], rtol=1e-13, err_msg=msg)