import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose, assert_equal
from statsmodels.regression.linear_model import GLSAR
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata
Testing GLSAR against STATA

Created on Wed May 30 09:25:24 2012

Author: Josef Perktold
