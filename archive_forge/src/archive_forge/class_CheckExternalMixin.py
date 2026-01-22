import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
class CheckExternalMixin:

    @classmethod
    def get_descriptives(cls, ddof=0):
        cls.descriptive = DescrStatsW(cls.data, cls.weights, ddof)

    @classmethod
    def save_data(cls, fname='data.csv'):
        df = pd.DataFrame(index=np.arange(len(cls.weights)))
        df['weights'] = cls.weights
        if cls.data.ndim == 1:
            df['data1'] = cls.data
        else:
            for k in range(cls.data.shape[1]):
                df['data%d' % (k + 1)] = cls.data[:, k]
        df.to_csv(fname)

    def test_mean(self):
        mn = self.descriptive.mean
        assert_allclose(mn, self.mean, rtol=0.0001)

    def test_sum(self):
        sm = self.descriptive.sum
        assert_allclose(sm, self.sum, rtol=0.0001)

    def test_var(self):
        var = self.descriptive.var
        assert_allclose(var, self.var, rtol=0.0001)

    def test_std(self):
        std = self.descriptive.std
        assert_allclose(std, self.std, rtol=0.0001)

    def test_sem(self):
        if not hasattr(self, 'sem'):
            return
        sem = self.descriptive.std_mean
        assert_allclose(sem, self.sem, rtol=0.0001)

    def test_quantiles(self):
        quant = np.asarray(self.quantiles, dtype=np.float64)
        for return_pandas in (False, True):
            qtl = self.descriptive.quantile(self.quantile_probs, return_pandas=return_pandas)
            qtl = np.asarray(qtl, dtype=np.float64)
            assert_allclose(qtl, quant, rtol=0.0001)