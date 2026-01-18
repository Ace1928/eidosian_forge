import numpy as np
from statsmodels.multivariate.factor import Factor
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
import warnings

    # R code:
    r = 0.4
    p = 6
    ii = seq(0, p-1)
    ii = outer(ii, ii, "-")
    ii = abs(ii)
    cm = r^ii
    factanal(covmat=cm, factors=2)
    