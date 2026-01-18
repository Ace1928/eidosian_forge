import os
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.regression.linear_model import OLS, GLSAR
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata
import statsmodels.stats.sandwich_covariance as sw
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi

        Augmented regression for common factor test
        OLS, using observations 1959:3-2009:3 (T = 201)
        Dependent variable: ds_l_realinv

                           coefficient   std. error   t-ratio    p-value
          ---------------------------------------------------------------
          const            -10.9481      1.35807      -8.062    7.44e-014 ***
          ds_l_realgdp       4.28893     0.229459     18.69     2.40e-045 ***
          realint_1         -0.662644    0.334872     -1.979    0.0492    **
          ds_l_realinv_1    -0.108892    0.0715042    -1.523    0.1294
          ds_l_realgdp_1     0.660443    0.390372      1.692    0.0923    *
          realint_2          0.0769695   0.341527      0.2254   0.8219

          Sum of squared residuals = 22432.8

        Test of common factor restriction

          Test statistic: F(2, 195) = 0.426391, with p-value = 0.653468
        