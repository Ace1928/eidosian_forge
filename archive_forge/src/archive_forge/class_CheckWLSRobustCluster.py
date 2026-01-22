import numpy as np
from numpy.testing import (
import pytest
from scipy import stats
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS, WLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.sm_exceptions import InvalidTestWarning
from statsmodels.tools.tools import add_constant
from .results import (
class CheckWLSRobustCluster(CheckOLSRobust):

    @classmethod
    def setup_class(cls):
        from statsmodels.datasets import grunfeld
        dtapa = grunfeld.data.load_pandas()
        dtapa_endog = dtapa.endog[:200]
        dtapa_exog = dtapa.exog[:200]
        exog = add_constant(dtapa_exog[['value', 'capital']], prepend=False)
        cls.res1 = WLS(dtapa_endog, exog, weights=1 / dtapa_exog['value']).fit()
        firm_names, firm_id = np.unique(np.asarray(dtapa_exog[['firm']], 'S20'), return_inverse=True)
        cls.groups = firm_id
        time = np.require(dtapa_exog[['year']], requirements='W')
        time -= time.min()
        cls.time = np.squeeze(time).astype(int)
        cls.tidx = [(i * 20, 20 * (i + 1)) for i in range(10)]