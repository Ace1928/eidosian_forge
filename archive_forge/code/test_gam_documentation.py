from statsmodels.compat.python import lrange
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats
import pytest
from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.sandbox.gam import Model as GAM #?
from statsmodels.genmod.families import family, links
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS
Tests for gam.AdditiveModel and GAM with Polynomials compared to OLS and GLM


Created on Sat Nov 05 14:16:07 2011

Author: Josef Perktold
License: BSD


Notes
-----

TODO: TestGAMGamma: has test failure (GLM looks good),
        adding log-link did not help
        resolved: gamma does not fail anymore after tightening the
                  convergence criterium (rtol=1e-6)
TODO: TestGAMNegativeBinomial: rvs generation does not work,
        nbinom needs 2 parameters
TODO: TestGAMGaussianLogLink: test failure,
        but maybe precision issue, not completely off

        but something is wrong, either the testcase or with the link
        >>> tt3.__class__
        <class '__main__.TestGAMGaussianLogLink'>
        >>> tt3.res2.mu_pred.mean()
        3.5616368292650766
        >>> tt3.res1.mu_pred.mean()
        3.6144278964707679
        >>> tt3.mu_true.mean()
        34.821904835958122
        >>>
        >>> tt3.y_true.mean()
        2.685225067611543
        >>> tt3.res1.y_pred.mean()
        0.52991541684645616
        >>> tt3.res2.y_pred.mean()
        0.44626406889363229



one possible change
~~~~~~~~~~~~~~~~~~~
add average, integral based tests, instead of or additional to sup
    * for example mean squared error for mu and eta (predict, fittedvalues)
      or mean absolute error, what's the scale for this? required precision?
    * this will also work for real non-parametric tests

example: Gamma looks good in average bias and average RMSE (RMISE)

>>> tt3 = _estGAMGamma()
>>> np.mean((tt3.res2.mu_pred - tt3.mu_true))/tt3.mu_true.mean()
-0.0051829977497423706
>>> np.mean((tt3.res2.y_pred - tt3.y_true))/tt3.y_true.mean()
0.00015255264651864049
>>> np.mean((tt3.res1.y_pred - tt3.y_true))/tt3.y_true.mean()
0.00015255538823786711
>>> np.mean((tt3.res1.mu_pred - tt3.mu_true))/tt3.mu_true.mean()
-0.0051937668989744494
>>> np.sqrt(np.mean((tt3.res1.mu_pred - tt3.mu_true)**2))/tt3.mu_true.mean()
0.022946118520401692
>>> np.sqrt(np.mean((tt3.res2.mu_pred - tt3.mu_true)**2))/tt3.mu_true.mean()
0.022953913332599746
>>> maxabs = lambda x: np.max(np.abs(x))
>>> maxabs((tt3.res1.mu_pred - tt3.mu_true))/tt3.mu_true.mean()
0.079540546242707733
>>> maxabs((tt3.res2.mu_pred - tt3.mu_true))/tt3.mu_true.mean()
0.079578857986784574
>>> maxabs((tt3.res2.y_pred - tt3.y_true))/tt3.y_true.mean()
0.016282852522951426
>>> maxabs((tt3.res1.y_pred - tt3.y_true))/tt3.y_true.mean()
0.016288391235613865



