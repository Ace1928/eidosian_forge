import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (

Tests for exponential smoothing models

Notes
-----

These tests are primarily against the `fpp` functions `ses`, `holt`, and `hw`
and against the `forecast` function `ets`. There are a couple of details about
how these packages work that are relevant for the tests:

Trend smoothing parameterization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that `fpp` and `ets` use
different parameterizations for the trend smoothing parameter. Our
implementation in `statespace.exponential_smoothing` uses the same
parameterization as `ets`.

The `fpp` package follows Holt's recursive equations directly, in which the
trend updating is:

.. math::

    b_t = \beta^* (\ell_t - \ell_{t-1}) + (1 - \beta^*) b_{t-1}

In our implementation, state updating is done by the Kalman filter, in which
the trend updating equation is:

.. math::

    b_{t|t} = b_{t|t-1} + \beta (y_t - l_{t|t-1})

by rewriting the Kalman updating equation in the form of Holt's method, we
find that we must have :math:`\beta = \beta^* \alpha`. This is the same
parameterization used by `ets`, which does not use the Kalman fitler but
instead uses an innovations state space framework.

Loglikelihood
^^^^^^^^^^^^^

The `ets` package has a `loglik` output value, but it does not compute the
loglikelihood itself, but rather a version without the constant parameters. It
appears to compute:

.. math::

    -\frac{n}{2} \log \left (\sum_{t=1}^n \varepsilon_t^2 \right)

while the loglikelihood is:

.. math::

    -\frac{n}{2}
    \log \left (2 \pi e \frac{1}{n} \sum_{t=1}^n \varepsilon_t^2 \right)

See Hyndman et al. (2008), pages 68-69. In particular, the former equation -
which is the value returned by `ets` - is -0.5 times equation (5.3), since for
these models we have :math:`r(x_{t-1}) = 1`. The latter equation is the log
of the likelihood formula given at the top of page 69.

Confidence intervals
^^^^^^^^^^^^^^^^^^^^

The range of the confidence intervals depends on the estimated variance,
sigma^2. In our default, we concentrate this variance out of the loglikelihood
function, meaning that the default is to use the maximum likelihood estimate
for forecasting purposes. forecast::ets uses a degree-of-freedom-corrected
estimate of sigma^2, and so our default confidence bands will differ. To
correct for this in the tests, we set `concentrate_scale=False` and use the
estimated variance from forecast::ets.

TODO: may want to add a parameter allowing specification of the variance
      estimator.

Author: Chad Fulton
License: BSD-3
