from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
class CointRankResults:
    """A class for holding the results from testing the cointegration rank.

    Parameters
    ----------
    rank : int (0 <= `rank` <= `neqs`)
        The rank to choose according to the Johansen cointegration rank
        test.
    neqs : int
        Number of variables in the time series.
    test_stats : array_like (`rank` + 1 if `rank` < `neqs` else `rank`)
        A one-dimensional array-like object containing the test statistics of
        the conducted tests.
    crit_vals : array_like (`rank` +1 if `rank` < `neqs` else `rank`)
        A one-dimensional array-like object containing the critical values
        corresponding to the entries in the `test_stats` argument.
    method : str, {``"trace"``, ``"maxeig"``}, default: ``"trace"``
        If ``"trace"``, the trace test statistic is used. If ``"maxeig"``, the
        maximum eigenvalue test statistic is used.
    signif : float, {0.1, 0.05, 0.01}, default: 0.05
        The test's significance level.
    """

    def __init__(self, rank, neqs, test_stats, crit_vals, method='trace', signif=0.05):
        self.rank = rank
        self.neqs = neqs
        self.r_1 = [neqs if method == 'trace' else i + 1 for i in range(min(rank + 1, neqs))]
        self.test_stats = test_stats
        self.crit_vals = crit_vals
        self.method = method
        self.signif = signif

    def summary(self):
        headers = ['r_0', 'r_1', 'test statistic', 'critical value']
        title = 'Johansen cointegration test using ' + ('trace' if self.method == 'trace' else 'maximum eigenvalue') + f' test statistic with {self.signif:.0%}' + ' significance level'
        num_tests = min(self.rank, self.neqs - 1)
        data = [[i, self.r_1[i], self.test_stats[i], self.crit_vals[i]] for i in range(num_tests + 1)]
        data_fmt = {'data_fmts': ['%s', '%s', '%#0.4g', '%#0.4g'], 'data_aligns': 'r'}
        html_data_fmt = dict(data_fmt)
        html_data_fmt['data_fmts'] = ['<td>' + i + '</td>' for i in html_data_fmt['data_fmts']]
        return SimpleTable(data=data, headers=headers, title=title, txt_fmt=data_fmt, html_fmt=html_data_fmt, ltx_fmt=data_fmt)

    def __str__(self):
        return self.summary().as_text()