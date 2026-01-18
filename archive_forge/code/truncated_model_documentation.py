import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.distributions.discrete import (
from statsmodels.discrete.discrete_model import (
from statsmodels.tools.numdiff import approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from copy import deepcopy

        Predict response variable or other statistic given exogenous variables.

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        exog : ndarray, optional
            Explanatory variables for the main count model.
            If ``exog`` is None, then the data from the model will be used.
        exog_infl : ndarray, optional
            Explanatory variables for the zero-inflation model.
            ``exog_infl`` has to be provided if ``exog`` was provided unless
            ``exog_infl`` in the model is only a constant.
        offset : ndarray, optional
            Offset is added to the linear predictor of the mean function with
            coefficient equal to 1.
            Default is zero if exog is not None, and the model offset if exog
            is None.
        exposure : ndarray, optional
            Log(exposure) is added to the linear predictor with coefficient
            equal to 1. If exposure is specified, then it will be logged by
            the method. The user does not need to log it first.
            Default is one if exog is is not None, and it is the model exposure
            if exog is None.
        which : str (optional)
            Statitistic to predict. Default is 'mean'.

            - 'mean' : the conditional expectation of endog E(y | x)
            - 'mean-main' : mean parameter of truncated count model.
              Note, this is not the mean of the truncated distribution.
            - 'linear' : the linear predictor of the truncated count model.
            - 'var' : returns the estimated variance of endog implied by the
              model.
            - 'prob-main' : probability of selecting the main model which is
              the probability of observing a nonzero count P(y > 0 | x).
            - 'prob-zero' : probability of observing a zero count. P(y=0 | x).
              This is equal to is ``1 - prob-main``
            - 'prob-trunc' : probability of truncation of the truncated count
              model. This is the probability of observing a zero count implied
              by the truncation model.
            - 'mean-nonzero' : expected value conditional on having observation
              larger than zero, E(y | X, y>0)
            - 'prob' : probabilities of each count from 0 to max(endog), or
              for y_values if those are provided. This is a multivariate
              return (2-dim when predicting for several observations).

        y_values : array_like
            Values of the random variable endog at which pmf is evaluated.
            Only used if ``which="prob"``

        Returns
        -------
        predicted values

        Notes
        -----
        'prob-zero' / 'prob-trunc' is the ratio of probabilities of observing
        a zero count between hurdle model and the truncated count model.
        If this ratio is larger than one, then the hurdle model has an inflated
        number of zeros compared to the count model. If it is smaller than one,
        then the number of zeros is deflated.
        