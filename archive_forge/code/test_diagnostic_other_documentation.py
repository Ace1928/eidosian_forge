import numpy as np
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.stats._diagnostic_other import CMTNewey, CMTTauchen
import statsmodels.stats._diagnostic_other as diao
Unit tests for generic score/LM tests and conditional moment tests

Created on Mon Nov 17 08:44:06 2014

Author: Josef Perktold
License: BSD-3

