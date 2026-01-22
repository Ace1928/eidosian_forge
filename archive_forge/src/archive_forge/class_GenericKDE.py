import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels
class GenericKDE:
    """
    Base class for density estimation and regression KDE classes.
    """

    def _compute_bw(self, bw):
        """
        Computes the bandwidth of the data.

        Parameters
        ----------
        bw : {array_like, str}
            If array_like: user-specified bandwidth.
            If a string, should be one of:

                - cv_ml: cross validation maximum likelihood
                - normal_reference: normal reference rule of thumb
                - cv_ls: cross validation least squares

        Notes
        -----
        The default values for bw is 'normal_reference'.
        """
        if bw is None:
            bw = 'normal_reference'
        if not isinstance(bw, str):
            self._bw_method = 'user-specified'
            res = np.asarray(bw)
        else:
            self._bw_method = bw
            if bw == 'normal_reference':
                bwfunc = self._normal_reference
            elif bw == 'cv_ml':
                bwfunc = self._cv_ml
            else:
                bwfunc = self._cv_ls
            res = bwfunc()
        return res

    def _compute_dispersion(self, data):
        """
        Computes the measure of dispersion.

        The minimum of the standard deviation and interquartile range / 1.349

        Notes
        -----
        Reimplemented in `KernelReg`, because the first column of `data` has to
        be removed.

        References
        ----------
        See the user guide for the np package in R.
        In the notes on bwscaling option in npreg, npudens, npcdens there is
        a discussion on the measure of dispersion
        """
        return _compute_min_std_IQR(data)

    def _get_class_vars_type(self):
        """Helper method to be able to pass needed vars to _compute_subset.

        Needs to be implemented by subclasses."""
        pass

    def _compute_efficient(self, bw):
        """
        Computes the bandwidth by estimating the scaling factor (c)
        in n_res resamples of size ``n_sub`` (in `randomize` case), or by
        dividing ``nobs`` into as many ``n_sub`` blocks as needed (if
        `randomize` is False).

        References
        ----------
        See p.9 in socserv.mcmaster.ca/racine/np_faq.pdf
        """
        if bw is None:
            self._bw_method = 'normal_reference'
        if isinstance(bw, str):
            self._bw_method = bw
        else:
            self._bw_method = 'user-specified'
            return bw
        nobs = self.nobs
        n_sub = self.n_sub
        data = copy.deepcopy(self.data)
        n_cvars = self.data_type.count('c')
        co = 4
        do = 4
        _, ix_ord, ix_unord = _get_type_pos(self.data_type)
        if self.randomize:
            bounds = [None] * self.n_res
        else:
            bounds = [(i * n_sub, (i + 1) * n_sub) for i in range(nobs // n_sub)]
            if nobs % n_sub > 0:
                bounds.append((nobs - nobs % n_sub, nobs))
        n_blocks = self.n_res if self.randomize else len(bounds)
        sample_scale = np.empty((n_blocks, self.k_vars))
        only_bw = np.empty((n_blocks, self.k_vars))
        class_type, class_vars = self._get_class_vars_type()
        if has_joblib:
            res = joblib.Parallel(n_jobs=self.n_jobs)((joblib.delayed(_compute_subset)(class_type, data, bw, co, do, n_cvars, ix_ord, ix_unord, n_sub, class_vars, self.randomize, bounds[i]) for i in range(n_blocks)))
        else:
            res = []
            for i in range(n_blocks):
                res.append(_compute_subset(class_type, data, bw, co, do, n_cvars, ix_ord, ix_unord, n_sub, class_vars, self.randomize, bounds[i]))
        for i in range(n_blocks):
            sample_scale[i, :] = res[i][0]
            only_bw[i, :] = res[i][1]
        s = self._compute_dispersion(data)
        order_func = np.median if self.return_median else np.mean
        m_scale = order_func(sample_scale, axis=0)
        bw = m_scale * s * nobs ** (-1.0 / (n_cvars + co))
        bw[ix_ord] = m_scale[ix_ord] * nobs ** (-2.0 / (n_cvars + do))
        bw[ix_unord] = m_scale[ix_unord] * nobs ** (-2.0 / (n_cvars + do))
        if self.return_only_bw:
            bw = np.median(only_bw, axis=0)
        return bw

    def _set_defaults(self, defaults):
        """Sets the default values for the efficient estimation"""
        self.n_res = defaults.n_res
        self.n_sub = defaults.n_sub
        self.randomize = defaults.randomize
        self.return_median = defaults.return_median
        self.efficient = defaults.efficient
        self.return_only_bw = defaults.return_only_bw
        self.n_jobs = defaults.n_jobs

    def _normal_reference(self):
        """
        Returns Scott's normal reference rule of thumb bandwidth parameter.

        Notes
        -----
        See p.13 in [2] for an example and discussion.  The formula for the
        bandwidth is

        .. math:: h = 1.06n^{-1/(4+q)}

        where ``n`` is the number of observations and ``q`` is the number of
        variables.
        """
        X = np.std(self.data, axis=0)
        return 1.06 * X * self.nobs ** (-1.0 / (4 + self.data.shape[1]))

    def _set_bw_bounds(self, bw):
        """
        Sets bandwidth lower bound to effectively zero )1e-10), and for
        discrete values upper bound to 1.
        """
        bw[bw < 0] = 1e-10
        _, ix_ord, ix_unord = _get_type_pos(self.data_type)
        bw[ix_ord] = np.minimum(bw[ix_ord], 1.0)
        bw[ix_unord] = np.minimum(bw[ix_unord], 1.0)
        return bw

    def _cv_ml(self):
        """
        Returns the cross validation maximum likelihood bandwidth parameter.

        Notes
        -----
        For more details see p.16, 18, 27 in Ref. [1] (see module docstring).

        Returns the bandwidth estimate that maximizes the leave-out-out
        likelihood.  The leave-one-out log likelihood function is:

        .. math:: \\ln L=\\sum_{i=1}^{n}\\ln f_{-i}(X_{i})

        The leave-one-out kernel estimator of :math:`f_{-i}` is:

        .. math:: f_{-i}(X_{i})=\\frac{1}{(n-1)h}
                        \\sum_{j=1,j\\neq i}K_{h}(X_{i},X_{j})

        where :math:`K_{h}` represents the Generalized product kernel
        estimator:

        .. math:: K_{h}(X_{i},X_{j})=\\prod_{s=1}^
                        {q}h_{s}^{-1}k\\left(\\frac{X_{is}-X_{js}}{h_{s}}\\right)
        """
        h0 = self._normal_reference()
        bw = optimize.fmin(self.loo_likelihood, x0=h0, args=(np.log,), maxiter=1000.0, maxfun=1000.0, disp=0, xtol=0.001)
        bw = self._set_bw_bounds(bw)
        return bw

    def _cv_ls(self):
        """
        Returns the cross-validation least squares bandwidth parameter(s).

        Notes
        -----
        For more details see pp. 16, 27 in Ref. [1] (see module docstring).

        Returns the value of the bandwidth that maximizes the integrated mean
        square error between the estimated and actual distribution.  The
        integrated mean square error (IMSE) is given by:

        .. math:: \\int\\left[\\hat{f}(x)-f(x)\\right]^{2}dx

        This is the general formula for the IMSE.  The IMSE differs for
        conditional (``KDEMultivariateConditional``) and unconditional
        (``KDEMultivariate``) kernel density estimation.
        """
        h0 = self._normal_reference()
        bw = optimize.fmin(self.imse, x0=h0, maxiter=1000.0, maxfun=1000.0, disp=0, xtol=0.001)
        bw = self._set_bw_bounds(bw)
        return bw

    def loo_likelihood(self):
        raise NotImplementedError