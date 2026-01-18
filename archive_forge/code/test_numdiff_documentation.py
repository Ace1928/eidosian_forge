import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
Testing numerical differentiation

Still some problems, with API (args tuple versus *args)
finite difference Hessian has some problems that I did not look at yet

Should Hessian also work per observation, if fun returns 2d

