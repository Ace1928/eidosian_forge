import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
def test_quantiles(self):
    quant = np.asarray(self.quantiles, dtype=np.float64)
    for return_pandas in (False, True):
        qtl = self.descriptive.quantile(self.quantile_probs, return_pandas=return_pandas)
        qtl = np.asarray(qtl, dtype=np.float64)
        assert_allclose(qtl, quant, rtol=0.0001)