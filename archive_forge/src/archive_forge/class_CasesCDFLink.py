import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less
from scipy import stats
import pytest
import statsmodels.genmod.families as families
from statsmodels.tools import numdiff as nd
class CasesCDFLink:
    link_pairs = [(links.CDFLink(dbn=stats.gumbel_l), links.CLogLog()), (links.CDFLink(dbn=stats.gumbel_r), links.LogLog()), (links.CDFLink(dbn=stats.norm), links.Probit()), (links.CDFLink(dbn=stats.logistic), links.Logit()), (links.CDFLink(dbn=stats.t(1)), links.Cauchy()), (MyCLogLog(), links.CLogLog())]
    methods = ['__call__', 'deriv', 'inverse', 'inverse_deriv', 'deriv2', 'inverse_deriv2']
    p = np.linspace(0, 1, 6)
    eps = 0.001
    p = np.clip(p, eps, 1 - eps)