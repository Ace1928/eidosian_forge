import inspect
import warnings
import numpy as np
from scipy import special, stats
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
from . import links as L, varfuncs as V
def _resid_dev(self, endog, mu):
    """
        Tweedie deviance residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable.
        mu : ndarray
            The inverse of the link function at the linear predicted values.

        Returns
        -------
        resid_dev : float
            Deviance residuals as defined below.

        Notes
        -----
        When :math:`p = 1`,

        .. math::

            dev_i = \\mu_i

        when :math:`endog_i = 0` and

        .. math::

            dev_i = endog_i * \\log(endog_i / \\mu_i) + (\\mu_i - endog_i)

        otherwise.

        When :math:`p = 2`,

        .. math::

            dev_i =  (endog_i - \\mu_i) / \\mu_i - \\log(endog_i / \\mu_i)

        For all other p,

        .. math::

            dev_i = endog_i^{2 - p} / ((1 - p) * (2 - p)) -
                    endog_i * \\mu_i^{1 - p} / (1 - p) + \\mu_i^{2 - p} /
                    (2 - p)

        The deviance residual is then

        .. math::

            resid\\_dev_i = 2 * dev_i
        """
    p = self.var_power
    if p == 1:
        dev = np.where(endog == 0, mu, endog * np.log(endog / mu) + (mu - endog))
    elif p == 2:
        endog1 = self._clean(endog)
        dev = (endog - mu) / mu - np.log(endog1 / mu)
    else:
        dev = endog ** (2 - p) / ((1 - p) * (2 - p)) - endog * mu ** (1 - p) / (1 - p) + mu ** (2 - p) / (2 - p)
    return 2 * dev