import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.stats.sandwich_covariance as sw
Tests for sandwich robust covariance estimation

see also in regression for cov_hac compared to Gretl and
sandbox.panel test_random_panel for comparing cov_cluster, cov_hac_panel and
cov_white

Created on Sat Dec 17 08:39:16 2011

Author: Josef Perktold
