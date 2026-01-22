from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
class MNLogit(MultinomialModel):
    __doc__ = '\n    Multinomial Logit Model\n\n    Parameters\n    ----------\n    endog : array_like\n        `endog` is an 1-d vector of the endogenous response.  `endog` can\n        contain strings, ints, or floats or may be a pandas Categorical Series.\n        Note that if it contains strings, every distinct string will be a\n        category.  No stripping of whitespace is done.\n    exog : array_like\n        A nobs x k array where `nobs` is the number of observations and `k`\n        is the number of regressors. An intercept is not included by default\n        and should be added by the user. See `statsmodels.tools.add_constant`.\n    {extra_params}\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    J : float\n        The number of choices for the endogenous variable. Note that this\n        is zero-indexed.\n    K : float\n        The actual number of parameters for the exogenous design.  Includes\n        the constant if the design has one.\n    names : dict\n        A dictionary mapping the column number in `wendog` to the variables\n        in `endog`.\n    wendog : ndarray\n        An n x j array where j is the number of unique categories in `endog`.\n        Each column of j is a dummy variable indicating the category of\n        each observation. See `names` for a dictionary mapping each column to\n        its category.\n\n    Notes\n    -----\n    See developer notes for further information on `MNLogit` internals.\n    '.format(extra_params=base._missing_param_doc + _check_rank_doc)

    def __init__(self, endog, exog, check_rank=True, **kwargs):
        super().__init__(endog, exog, check_rank=check_rank, **kwargs)
        yname = self.endog_names
        ynames = self._ynames_map
        ynames = MultinomialResults._maybe_convert_ynames_int(ynames)
        ynames = [ynames[key] for key in range(int(self.J))]
        idx = MultiIndex.from_product((ynames[1:], self.data.xnames), names=(yname, None))
        self.data.cov_names = idx

    def pdf(self, eXB):
        """
        NotImplemented
        """
        raise NotImplementedError

    def cdf(self, X):
        """
        Multinomial logit cumulative distribution function.

        Parameters
        ----------
        X : ndarray
            The linear predictor of the model XB.

        Returns
        -------
        cdf : ndarray
            The cdf evaluated at `X`.

        Notes
        -----
        In the multinomial logit model.
        .. math:: \\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}
        """
        eXB = np.column_stack((np.ones(len(X)), np.exp(X)))
        return eXB / eXB.sum(1)[:, None]

    def loglike(self, params):
        """
        Log-likelihood of the multinomial logit model.

        Parameters
        ----------
        params : array_like
            The parameters of the multinomial logit model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        -----
        .. math::

           \\ln L=\\sum_{i=1}^{n}\\sum_{j=0}^{J}d_{ij}\\ln
           \\left(\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}
           {\\sum_{k=0}^{J}
           \\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)

        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.
        """
        params = params.reshape(self.K, -1, order='F')
        d = self.wendog
        logprob = np.log(self.cdf(np.dot(self.exog, params)))
        return np.sum(d * logprob)

    def loglikeobs(self, params):
        """
        Log-likelihood of the multinomial logit model for each observation.

        Parameters
        ----------
        params : array_like
            The parameters of the multinomial logit model.

        Returns
        -------
        loglike : array_like
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        -----
        .. math::

           \\ln L_{i}=\\sum_{j=0}^{J}d_{ij}\\ln
           \\left(\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}
           {\\sum_{k=0}^{J}
           \\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)

        for observations :math:`i=1,...,n`

        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.
        """
        params = params.reshape(self.K, -1, order='F')
        d = self.wendog
        logprob = np.log(self.cdf(np.dot(self.exog, params)))
        return d * logprob

    def score(self, params):
        """
        Score matrix for multinomial logit model log-likelihood

        Parameters
        ----------
        params : ndarray
            The parameters of the multinomial logit model.

        Returns
        -------
        score : ndarray, (K * (J-1),)
            The 2-d score vector, i.e. the first derivative of the
            loglikelihood function, of the multinomial logit model evaluated at
            `params`.

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta_{j}}=\\sum_{i}\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}

        for :math:`j=1,...,J`

        In the multinomial model the score matrix is K x J-1 but is returned
        as a flattened array to work with the solvers.
        """
        params = params.reshape(self.K, -1, order='F')
        firstterm = self.wendog[:, 1:] - self.cdf(np.dot(self.exog, params))[:, 1:]
        return np.dot(firstterm.T, self.exog).flatten()

    def loglike_and_score(self, params):
        """
        Returns log likelihood and score, efficiently reusing calculations.

        Note that both of these returned quantities will need to be negated
        before being minimized by the maximum likelihood fitting machinery.
        """
        params = params.reshape(self.K, -1, order='F')
        cdf_dot_exog_params = self.cdf(np.dot(self.exog, params))
        loglike_value = np.sum(self.wendog * np.log(cdf_dot_exog_params))
        firstterm = self.wendog[:, 1:] - cdf_dot_exog_params[:, 1:]
        score_array = np.dot(firstterm.T, self.exog).flatten()
        return (loglike_value, score_array)

    def score_obs(self, params):
        """
        Jacobian matrix for multinomial logit model log-likelihood

        Parameters
        ----------
        params : ndarray
            The parameters of the multinomial logit model.

        Returns
        -------
        jac : array_like
            The derivative of the loglikelihood for each observation evaluated
            at `params` .

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta_{j}}=\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}

        for :math:`j=1,...,J`, for observations :math:`i=1,...,n`

        In the multinomial model the score vector is K x (J-1) but is returned
        as a flattened array. The Jacobian has the observations in rows and
        the flattened array of derivatives in columns.
        """
        params = params.reshape(self.K, -1, order='F')
        firstterm = self.wendog[:, 1:] - self.cdf(np.dot(self.exog, params))[:, 1:]
        return (firstterm[:, :, None] * self.exog[:, None, :]).reshape(self.exog.shape[0], -1)

    def hessian(self, params):
        """
        Multinomial logit Hessian matrix of the log-likelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (J*K, J*K)
            The Hessian, second derivative of loglikelihood function with
            respect to the flattened parameters, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta_{j}\\partial\\beta_{l}}=-\\sum_{i=1}^{n}\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\left[\\boldsymbol{1}\\left(j=l\\right)-\\frac{\\exp\\left(\\beta_{l}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right]x_{i}x_{l}^{\\prime}

        where
        :math:`\\boldsymbol{1}\\left(j=l\\right)` equals 1 if `j` = `l` and 0
        otherwise.

        The actual Hessian matrix has J**2 * K x K elements. Our Hessian
        is reshaped to be square (J*K, J*K) so that the solvers can use it.

        This implementation does not take advantage of the symmetry of
        the Hessian and could probably be refactored for speed.
        """
        params = params.reshape(self.K, -1, order='F')
        X = self.exog
        pr = self.cdf(np.dot(X, params))
        partials = []
        J = self.J
        K = self.K
        for i in range(J - 1):
            for j in range(J - 1):
                if i == j:
                    partials.append(-np.dot(((pr[:, i + 1] * (1 - pr[:, j + 1]))[:, None] * X).T, X))
                else:
                    partials.append(-np.dot(((pr[:, i + 1] * -pr[:, j + 1])[:, None] * X).T, X))
        H = np.array(partials)
        H = np.transpose(H.reshape(J - 1, J - 1, K, K), (0, 2, 1, 3)).reshape((J - 1) * K, (J - 1) * K)
        return H