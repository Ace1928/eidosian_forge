import inspect
import warnings
import numpy as np
from scipy import special, stats
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
from . import links as L, varfuncs as V

        The Anscombe residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable
        mu : ndarray
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float, optional
            An optional argument to divide the residuals by sqrt(scale).
            The default is 1.

        Returns
        -------
        resid_anscombe : ndarray
            The Anscombe residuals as defined below.

        Notes
        -----
        When :math:`p = 3`, then

        .. math::

            resid\_anscombe_i = \log(endog_i / \mu_i) / \sqrt{\mu_i * scale} *
            \sqrt(var\_weights)

        Otherwise,

        .. math::

            c = (3 - p) / 3

        .. math::

            resid\_anscombe_i = (1 / c) * (endog_i^c - \mu_i^c) / \mu_i^{p / 6}
            / \sqrt{scale} * \sqrt(var\_weights)
        