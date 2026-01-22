import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
class CheckStratifiedMixin:

    @classmethod
    def initialize(cls, tables, use_arr=False):
        tables1 = tables if not use_arr else np.dstack(tables)
        cls.rslt = ctab.StratifiedTable(tables1)
        cls.rslt_0 = ctab.StratifiedTable(tables, shift_zeros=True)
        tables_pandas = [pd.DataFrame(x) for x in tables]
        cls.rslt_pandas = ctab.StratifiedTable(tables_pandas)

    def test_oddsratio_pooled(self):
        assert_allclose(self.rslt.oddsratio_pooled, self.oddsratio_pooled, rtol=0.0001, atol=0.0001)

    def test_logodds_pooled(self):
        assert_allclose(self.rslt.logodds_pooled, self.logodds_pooled, rtol=0.0001, atol=0.0001)

    def test_null_odds(self):
        rslt = self.rslt.test_null_odds(correction=True)
        assert_allclose(rslt.statistic, self.mh_stat, rtol=0.0001, atol=1e-05)
        assert_allclose(rslt.pvalue, self.mh_pvalue, rtol=0.0001, atol=0.0001)

    def test_oddsratio_pooled_confint(self):
        lcb, ucb = self.rslt.oddsratio_pooled_confint()
        assert_allclose(lcb, self.or_lcb, rtol=0.0001, atol=0.0001)
        assert_allclose(ucb, self.or_ucb, rtol=0.0001, atol=0.0001)

    def test_logodds_pooled_confint(self):
        lcb, ucb = self.rslt.logodds_pooled_confint()
        assert_allclose(lcb, np.log(self.or_lcb), rtol=0.0001, atol=0.0001)
        assert_allclose(ucb, np.log(self.or_ucb), rtol=0.0001, atol=0.0001)

    def test_equal_odds(self):
        if not hasattr(self, 'or_homog'):
            return
        rslt = self.rslt.test_equal_odds(adjust=False)
        assert_allclose(rslt.statistic, self.or_homog, rtol=0.0001, atol=0.0001)
        assert_allclose(rslt.pvalue, self.or_homog_p, rtol=0.0001, atol=0.0001)
        rslt = self.rslt.test_equal_odds(adjust=True)
        assert_allclose(rslt.statistic, self.or_homog_adj, rtol=0.0001, atol=0.0001)
        assert_allclose(rslt.pvalue, self.or_homog_adj_p, rtol=0.0001, atol=0.0001)

    def test_pandas(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            assert_equal(self.rslt.summary().as_text(), self.rslt_pandas.summary().as_text())

    def test_from_data(self):
        np.random.seed(241)
        df = pd.DataFrame(index=range(100), columns=('v1', 'v2', 'strat'))
        df['v1'] = np.random.randint(0, 2, 100)
        df['v2'] = np.random.randint(0, 2, 100)
        df['strat'] = np.kron(np.arange(10), np.ones(10))
        tables = []
        for k in range(10):
            ii = np.arange(10 * k, 10 * (k + 1))
            tables.append(pd.crosstab(df.loc[ii, 'v1'], df.loc[ii, 'v2']))
        rslt1 = ctab.StratifiedTable(tables)
        rslt2 = ctab.StratifiedTable.from_data('v1', 'v2', 'strat', df)
        assert_equal(rslt1.summary().as_text(), rslt2.summary().as_text())