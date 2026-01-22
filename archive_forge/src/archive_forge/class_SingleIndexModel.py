import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from statsmodels.nonparametric.api import KDEMultivariate, KernelReg
from statsmodels.nonparametric._kernel_base import \
class SingleIndexModel(KernelReg):
    """
    Single index semiparametric model ``y = g(X * b) + e``.

    Parameters
    ----------
    endog : array_like
        The dependent variable
    exog : array_like
        The independent variable(s)
    var_type : str
        The type of variables in X:

            - c: continuous
            - o: ordered
            - u: unordered

    Attributes
    ----------
    b : array_like
        The linear coefficients b (betas)
    bw : array_like
        Bandwidths

    Methods
    -------
    fit(): Computes the fitted values ``E[Y|X] = g(X * b)``
           and the marginal effects ``dY/dX``.

    References
    ----------
    See chapter on semiparametric models in [1]

    Notes
    -----
    This model resembles the binary choice models. The user knows
    that X and b interact linearly, but ``g(X * b)`` is unknown.
    In the parametric binary choice models the user usually assumes
    some distribution of g() such as normal or logistic.
    """

    def __init__(self, endog, exog, var_type):
        self.var_type = var_type
        self.K = len(var_type)
        self.var_type = self.var_type[0]
        self.endog = _adjust_shape(endog, 1)
        self.exog = _adjust_shape(exog, self.K)
        self.nobs = np.shape(self.exog)[0]
        self.data_type = self.var_type
        self.ckertype = 'gaussian'
        self.okertype = 'wangryzin'
        self.ukertype = 'aitchisonaitken'
        self.func = self._est_loc_linear
        self.b, self.bw = self._est_b_bw()

    def _est_b_bw(self):
        params0 = np.random.uniform(size=(self.K + 1,))
        b_bw = optimize.fmin(self.cv_loo, params0, disp=0)
        b = b_bw[0:self.K]
        bw = b_bw[self.K:]
        bw = self._set_bw_bounds(bw)
        return (b, bw)

    def cv_loo(self, params):
        params = np.asarray(params)
        b = params[0:self.K]
        bw = params[self.K:]
        LOO_X = LeaveOneOut(self.exog)
        LOO_Y = LeaveOneOut(self.endog).__iter__()
        L = 0
        for i, X_not_i in enumerate(LOO_X):
            Y = next(LOO_Y)
            G = self.func(bw, endog=Y, exog=-np.dot(X_not_i, b)[:, None], data_predict=-np.dot(self.exog[i:i + 1, :], b))[0]
            L += (self.endog[i] - G) ** 2
        return L / self.nobs

    def fit(self, data_predict=None):
        if data_predict is None:
            data_predict = self.exog
        else:
            data_predict = _adjust_shape(data_predict, self.K)
        N_data_predict = np.shape(data_predict)[0]
        mean = np.empty((N_data_predict,))
        mfx = np.empty((N_data_predict, self.K))
        for i in range(N_data_predict):
            mean_mfx = self.func(self.bw, self.endog, np.dot(self.exog, self.b)[:, None], data_predict=np.dot(data_predict[i:i + 1, :], self.b))
            mean[i] = mean_mfx[0]
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c
        return (mean, mfx)

    def __repr__(self):
        """Provide something sane to print."""
        repr = 'Single Index Model \n'
        repr += 'Number of variables: K = ' + str(self.K) + '\n'
        repr += 'Number of samples:   nobs = ' + str(self.nobs) + '\n'
        repr += 'Variable types:      ' + self.var_type + '\n'
        repr += 'BW selection method: cv_ls' + '\n'
        repr += 'Estimator type: local constant' + '\n'
        return repr