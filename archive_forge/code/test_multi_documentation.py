import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
Tests for multipletests and fdr pvalue corrections

Author : Josef Perktold


['b', 's', 'sh', 'hs', 'h', 'fdr_i', 'fdr_n', 'fdr_tsbh']
are tested against R:multtest

'hommel' is tested against R stats p_adjust (not available in multtest

'fdr_gbs', 'fdr_2sbky' I did not find them in R, currently tested for
    consistency only

