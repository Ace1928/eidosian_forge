import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from statsmodels.nonparametric.api import KDEMultivariate, KernelReg
from statsmodels.nonparametric._kernel_base import \
class SemiLinear(KernelReg):
    """
    Semiparametric partially linear model, ``Y = Xb + g(Z) + e``.

    Parameters
    ----------
    endog : array_like
        The dependent variable
    exog : array_like
        The linear component in the regression
    exog_nonparametric : array_like
        The nonparametric component in the regression
    var_type : str
        The type of the variables in the nonparametric component;

            - c: continuous
            - o: ordered
            - u: unordered

    k_linear : int
        The number of variables that comprise the linear component.

    Attributes
    ----------
    bw : array_like
        Bandwidths for the nonparametric component exog_nonparametric
    b : array_like
        Coefficients in the linear component
    nobs : int
        The number of observations.
    k_linear : int
        The number of variables that comprise the linear component.

    Methods
    -------
    fit
        Returns the fitted mean and marginal effects dy/dz

    Notes
    -----
    This model uses only the local constant regression estimator

    References
    ----------
    See chapter on Semiparametric Models in [1]
    """

    def __init__(self, endog, exog, exog_nonparametric, var_type, k_linear):
        self.endog = _adjust_shape(endog, 1)
        self.exog = _adjust_shape(exog, k_linear)
        self.K = len(var_type)
        self.exog_nonparametric = _adjust_shape(exog_nonparametric, self.K)
        self.k_linear = k_linear
        self.nobs = np.shape(self.exog)[0]
        self.var_type = var_type
        self.data_type = self.var_type
        self.ckertype = 'gaussian'
        self.okertype = 'wangryzin'
        self.ukertype = 'aitchisonaitken'
        self.func = self._est_loc_linear
        self.b, self.bw = self._est_b_bw()

    def _est_b_bw(self):
        """
        Computes the (beta) coefficients and the bandwidths.

        Minimizes ``cv_loo`` with respect to ``b`` and ``bw``.
        """
        params0 = np.random.uniform(size=(self.k_linear + self.K,))
        b_bw = optimize.fmin(self.cv_loo, params0, disp=0)
        b = b_bw[0:self.k_linear]
        bw = b_bw[self.k_linear:]
        return (b, bw)

    def cv_loo(self, params):
        """
        Similar to the cross validation leave-one-out estimator.

        Modified to reflect the linear components.

        Parameters
        ----------
        params : array_like
            Vector consisting of the coefficients (b) and the bandwidths (bw).
            The first ``k_linear`` elements are the coefficients.

        Returns
        -------
        L : float
            The value of the objective function

        References
        ----------
        See p.254 in [1]
        """
        params = np.asarray(params)
        b = params[0:self.k_linear]
        bw = params[self.k_linear:]
        LOO_X = LeaveOneOut(self.exog)
        LOO_Y = LeaveOneOut(self.endog).__iter__()
        LOO_Z = LeaveOneOut(self.exog_nonparametric).__iter__()
        Xb = np.dot(self.exog, b)[:, None]
        L = 0
        for ii, X_not_i in enumerate(LOO_X):
            Y = next(LOO_Y)
            Z = next(LOO_Z)
            Xb_j = np.dot(X_not_i, b)[:, None]
            Yx = Y - Xb_j
            G = self.func(bw, endog=Yx, exog=-Z, data_predict=-self.exog_nonparametric[ii, :])[0]
            lt = Xb[ii, :]
            L += (self.endog[ii] - lt - G) ** 2
        return L

    def fit(self, exog_predict=None, exog_nonparametric_predict=None):
        """Computes fitted values and marginal effects"""
        if exog_predict is None:
            exog_predict = self.exog
        else:
            exog_predict = _adjust_shape(exog_predict, self.k_linear)
        if exog_nonparametric_predict is None:
            exog_nonparametric_predict = self.exog_nonparametric
        else:
            exog_nonparametric_predict = _adjust_shape(exog_nonparametric_predict, self.K)
        N_data_predict = np.shape(exog_nonparametric_predict)[0]
        mean = np.empty((N_data_predict,))
        mfx = np.empty((N_data_predict, self.K))
        Y = self.endog - np.dot(exog_predict, self.b)[:, None]
        for i in range(N_data_predict):
            mean_mfx = self.func(self.bw, Y, self.exog_nonparametric, data_predict=exog_nonparametric_predict[i, :])
            mean[i] = mean_mfx[0]
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c
        return (mean, mfx)

    def __repr__(self):
        """Provide something sane to print."""
        repr = 'Semiparamatric Partially Linear Model \n'
        repr += 'Number of variables: K = ' + str(self.K) + '\n'
        repr += 'Number of samples:   N = ' + str(self.nobs) + '\n'
        repr += 'Variable types:      ' + self.var_type + '\n'
        repr += 'BW selection method: cv_ls' + '\n'
        repr += 'Estimator type: local constant' + '\n'
        return repr