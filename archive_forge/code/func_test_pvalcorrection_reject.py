import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
@pytest.mark.parametrize('alpha', [0.01, 0.05, 0.1])
@pytest.mark.parametrize('method', ['b', 's', 'sh', 'hs', 'h', 'hommel', 'fdr_i', 'fdr_n', 'fdr_tsbky', 'fdr_tsbh', 'fdr_gbs'])
@pytest.mark.parametrize('ii', list(range(11)))
def test_pvalcorrection_reject(alpha, method, ii):
    pval1 = np.hstack((np.linspace(0.0001, 0.01, ii), np.linspace(0.05001, 0.11, 10 - ii)))
    reject, pvalscorr = multipletests(pval1, alpha=alpha, method=method)[:2]
    msg = 'case %s %3.2f rejected:%d\npval_raw=%r\npvalscorr=%r' % (method, alpha, reject.sum(), pval1, pvalscorr)
    assert_equal(reject, pvalscorr <= alpha, err_msg=msg)