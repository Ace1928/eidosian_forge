import numpy as np
from . import kernels
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
class KDEMultivariateConditional(GenericKDE):
    """
    Conditional multivariate kernel density estimator.

    Calculates ``P(Y_1,Y_2,...Y_n | X_1,X_2...X_m) =
    P(X_1, X_2,...X_n, Y_1, Y_2,..., Y_m)/P(X_1, X_2,..., X_m)``.
    The conditional density is by definition the ratio of the two densities,
    see [1]_.

    Parameters
    ----------
    endog : list of ndarrays or 2-D ndarray
        The training data for the dependent variables, used to determine
        the bandwidth(s).  If a 2-D array, should be of shape
        (num_observations, num_variables).  If a list, each list element is a
        separate observation.
    exog : list of ndarrays or 2-D ndarray
        The training data for the independent variable; same shape as `endog`.
    dep_type : str
        The type of the dependent variables:

            c : Continuous
            u : Unordered (Discrete)
            o : Ordered (Discrete)

        The string should contain a type specifier for each variable, so for
        example ``dep_type='ccuo'``.
    indep_type : str
        The type of the independent variables; specified like `dep_type`.
    bw : array_like or str, optional
        If an array, it is a fixed user-specified bandwidth.  If a string,
        should be one of:

            - normal_reference: normal reference rule of thumb (default)
            - cv_ml: cross validation maximum likelihood
            - cv_ls: cross validation least squares

    defaults : Instance of class EstimatorSettings
        The default values for the efficient bandwidth estimation

    Attributes
    ----------
    bw : array_like
        The bandwidth parameters

    See Also
    --------
    KDEMultivariate

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Conditional_probability_distribution

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> nobs = 300
    >>> c1 = np.random.normal(size=(nobs,1))
    >>> c2 = np.random.normal(2,1,size=(nobs,1))

    >>> dens_c = sm.nonparametric.KDEMultivariateConditional(endog=[c1],
    ...     exog=[c2], dep_type='c', indep_type='c', bw='normal_reference')
    >>> dens_c.bw   # show computed bandwidth
    array([ 0.41223484,  0.40976931])
    """

    def __init__(self, endog, exog, dep_type, indep_type, bw, defaults=None):
        self.dep_type = dep_type
        self.indep_type = indep_type
        self.data_type = dep_type + indep_type
        self.k_dep = len(self.dep_type)
        self.k_indep = len(self.indep_type)
        self.endog = _adjust_shape(endog, self.k_dep)
        self.exog = _adjust_shape(exog, self.k_indep)
        self.nobs, self.k_dep = np.shape(self.endog)
        self.data = np.column_stack((self.endog, self.exog))
        self.k_vars = np.shape(self.data)[1]
        defaults = EstimatorSettings() if defaults is None else defaults
        self._set_defaults(defaults)
        if not self.efficient:
            self.bw = self._compute_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def __repr__(self):
        """Provide something sane to print."""
        rpr = 'KDEMultivariateConditional instance\n'
        rpr += 'Number of independent variables: k_indep = ' + str(self.k_indep) + '\n'
        rpr += 'Number of dependent variables: k_dep = ' + str(self.k_dep) + '\n'
        rpr += 'Number of observations: nobs = ' + str(self.nobs) + '\n'
        rpr += 'Independent variable types:      ' + self.indep_type + '\n'
        rpr += 'Dependent variable types:      ' + self.dep_type + '\n'
        rpr += 'BW selection method: ' + self._bw_method + '\n'
        return rpr

    def loo_likelihood(self, bw, func=lambda x: x):
        """
        Returns the leave-one-out conditional likelihood of the data.

        If `func` is not equal to the default, what's calculated is a function
        of the leave-one-out conditional likelihood.

        Parameters
        ----------
        bw : array_like
            The bandwidth parameter(s).
        func : callable, optional
            Function to transform the likelihood values (before summing); for
            the log likelihood, use ``func=np.log``.  Default is ``f(x) = x``.

        Returns
        -------
        L : float
            The value of the leave-one-out function for the data.

        Notes
        -----
        Similar to ``KDE.loo_likelihood`, but substitute ``f(y|x)=f(x,y)/f(x)``
        for ``f(x)``.
        """
        yLOO = LeaveOneOut(self.data)
        xLOO = LeaveOneOut(self.exog).__iter__()
        L = 0
        for i, Y_j in enumerate(yLOO):
            X_not_i = next(xLOO)
            f_yx = gpke(bw, data=-Y_j, data_predict=-self.data[i, :], var_type=self.dep_type + self.indep_type)
            f_x = gpke(bw[self.k_dep:], data=-X_not_i, data_predict=-self.exog[i, :], var_type=self.indep_type)
            f_i = f_yx / f_x
            L += func(f_i)
        return -L

    def pdf(self, endog_predict=None, exog_predict=None):
        """
        Evaluate the probability density function.

        Parameters
        ----------
        endog_predict : array_like, optional
            Evaluation data for the dependent variables.  If unspecified, the
            training data is used.
        exog_predict : array_like, optional
            Evaluation data for the independent variables.

        Returns
        -------
        pdf : array_like
            The value of the probability density at `endog_predict` and `exog_predict`.

        Notes
        -----
        The formula for the conditional probability density is:

        .. math:: f(y|x)=\\frac{f(x,y)}{f(x)}

        with

        .. math:: f(x)=\\prod_{s=1}^{q}h_{s}^{-1}k
                            \\left(\\frac{x_{is}-x_{js}}{h_{s}}\\right)

        where :math:`k` is the appropriate kernel for each variable.
        """
        if endog_predict is None:
            endog_predict = self.endog
        else:
            endog_predict = _adjust_shape(endog_predict, self.k_dep)
        if exog_predict is None:
            exog_predict = self.exog
        else:
            exog_predict = _adjust_shape(exog_predict, self.k_indep)
        pdf_est = []
        data_predict = np.column_stack((endog_predict, exog_predict))
        for i in range(np.shape(data_predict)[0]):
            f_yx = gpke(self.bw, data=self.data, data_predict=data_predict[i, :], var_type=self.dep_type + self.indep_type)
            f_x = gpke(self.bw[self.k_dep:], data=self.exog, data_predict=exog_predict[i, :], var_type=self.indep_type)
            pdf_est.append(f_yx / f_x)
        return np.squeeze(pdf_est)

    def cdf(self, endog_predict=None, exog_predict=None):
        """
        Cumulative distribution function for the conditional density.

        Parameters
        ----------
        endog_predict : array_like, optional
            The evaluation dependent variables at which the cdf is estimated.
            If not specified the training dependent variables are used.
        exog_predict : array_like, optional
            The evaluation independent variables at which the cdf is estimated.
            If not specified the training independent variables are used.

        Returns
        -------
        cdf_est : array_like
            The estimate of the cdf.

        Notes
        -----
        For more details on the estimation see [2]_, and p.181 in [1]_.

        The multivariate conditional CDF for mixed data (continuous and
        ordered/unordered discrete) is estimated by:

        .. math::

            F(y|x)=\\frac{n^{-1}\\sum_{i=1}^{n}G(\\frac{y-Y_{i}}{h_{0}}) W_{h}(X_{i},x)}{\\widehat{\\mu}(x)}

        where G() is the product kernel CDF estimator for the dependent (y)
        variable(s) and W() is the product kernel CDF estimator for the
        independent variable(s).

        References
        ----------
        .. [1] Racine, J., Li, Q. Nonparametric econometrics: theory and
                practice. Princeton University Press. (2007)
        .. [2] Liu, R., Yang, L. "Kernel estimation of multivariate cumulative
                    distribution function." Journal of Nonparametric
                    Statistics (2008)
        """
        if endog_predict is None:
            endog_predict = self.endog
        else:
            endog_predict = _adjust_shape(endog_predict, self.k_dep)
        if exog_predict is None:
            exog_predict = self.exog
        else:
            exog_predict = _adjust_shape(exog_predict, self.k_indep)
        N_data_predict = np.shape(exog_predict)[0]
        cdf_est = np.empty(N_data_predict)
        for i in range(N_data_predict):
            mu_x = gpke(self.bw[self.k_dep:], data=self.exog, data_predict=exog_predict[i, :], var_type=self.indep_type) / self.nobs
            mu_x = np.squeeze(mu_x)
            cdf_endog = gpke(self.bw[0:self.k_dep], data=self.endog, data_predict=endog_predict[i, :], var_type=self.dep_type, ckertype='gaussian_cdf', ukertype='aitchisonaitken_cdf', okertype='wangryzin_cdf', tosum=False)
            cdf_exog = gpke(self.bw[self.k_dep:], data=self.exog, data_predict=exog_predict[i, :], var_type=self.indep_type, tosum=False)
            S = (cdf_endog * cdf_exog).sum(axis=0)
            cdf_est[i] = S / (self.nobs * mu_x)
        return cdf_est

    def imse(self, bw):
        """
        The integrated mean square error for the conditional KDE.

        Parameters
        ----------
        bw : array_like
            The bandwidth parameter(s).

        Returns
        -------
        CV : float
            The cross-validation objective function.

        Notes
        -----
        For more details see pp. 156-166 in [1]_. For details on how to
        handle the mixed variable types see [2]_.

        The formula for the cross-validation objective function for mixed
        variable types is:

        .. math:: CV(h,\\lambda)=\\frac{1}{n}\\sum_{l=1}^{n}
            \\frac{G_{-l}(X_{l})}{\\left[\\mu_{-l}(X_{l})\\right]^{2}}-
            \\frac{2}{n}\\sum_{l=1}^{n}\\frac{f_{-l}(X_{l},Y_{l})}{\\mu_{-l}(X_{l})}

        where

        .. math:: G_{-l}(X_{l}) = n^{-2}\\sum_{i\\neq l}\\sum_{j\\neq l}
                        K_{X_{i},X_{l}} K_{X_{j},X_{l}}K_{Y_{i},Y_{j}}^{(2)}

        where :math:`K_{X_{i},X_{l}}` is the multivariate product kernel and
        :math:`\\mu_{-l}(X_{l})` is the leave-one-out estimator of the pdf.

        :math:`K_{Y_{i},Y_{j}}^{(2)}` is the convolution kernel.

        The value of the function is minimized by the ``_cv_ls`` method of the
        `GenericKDE` class to return the bw estimates that minimize the
        distance between the estimated and "true" probability density.

        References
        ----------
        .. [1] Racine, J., Li, Q. Nonparametric econometrics: theory and
                practice. Princeton University Press. (2007)
        .. [2] Racine, J., Li, Q. "Nonparametric Estimation of Distributions
                with Categorical and Continuous Data." Working Paper. (2000)
        """
        zLOO = LeaveOneOut(self.data)
        CV = 0
        nobs = float(self.nobs)
        expander = np.ones((self.nobs - 1, 1))
        for ii, Z in enumerate(zLOO):
            X = Z[:, self.k_dep:]
            Y = Z[:, :self.k_dep]
            Ye_L = np.kron(Y, expander)
            Ye_R = np.kron(expander, Y)
            Xe_L = np.kron(X, expander)
            Xe_R = np.kron(expander, X)
            K_Xi_Xl = gpke(bw[self.k_dep:], data=Xe_L, data_predict=self.exog[ii, :], var_type=self.indep_type, tosum=False)
            K_Xj_Xl = gpke(bw[self.k_dep:], data=Xe_R, data_predict=self.exog[ii, :], var_type=self.indep_type, tosum=False)
            K2_Yi_Yj = gpke(bw[0:self.k_dep], data=Ye_L, data_predict=Ye_R, var_type=self.dep_type, ckertype='gauss_convolution', okertype='wangryzin_convolution', ukertype='aitchisonaitken_convolution', tosum=False)
            G = (K_Xi_Xl * K_Xj_Xl * K2_Yi_Yj).sum() / nobs ** 2
            f_X_Y = gpke(bw, data=-Z, data_predict=-self.data[ii, :], var_type=self.dep_type + self.indep_type) / nobs
            m_x = gpke(bw[self.k_dep:], data=-X, data_predict=-self.exog[ii, :], var_type=self.indep_type) / nobs
            CV += G / m_x ** 2 - 2 * (f_X_Y / m_x)
        return CV / nobs

    def _get_class_vars_type(self):
        """Helper method to be able to pass needed vars to _compute_subset."""
        class_type = 'KDEMultivariateConditional'
        class_vars = (self.k_dep, self.dep_type, self.indep_type)
        return (class_type, class_vars)