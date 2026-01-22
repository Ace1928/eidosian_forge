import warnings
import numpy as np
import pandas as pd
from statsmodels.base import model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ConvergenceWarning
class SlicedInverseReg(_DimReductionRegression):
    """
    Sliced Inverse Regression (SIR)

    Parameters
    ----------
    endog : array_like (1d)
        The dependent variable
    exog : array_like (2d)
        The covariates

    References
    ----------
    KC Li (1991).  Sliced inverse regression for dimension reduction.
    JASA 86, 316-342.
    """

    def fit(self, slice_n=20, **kwargs):
        """
        Estimate the EDR space using Sliced Inverse Regression.

        Parameters
        ----------
        slice_n : int, optional
            Target number of observations per slice
        """
        if len(kwargs) > 0:
            msg = 'SIR.fit does not take any extra keyword arguments'
            warnings.warn(msg)
        n_slice = self.exog.shape[0] // slice_n
        self._prep(n_slice)
        mn = [z.mean(0) for z in self._split_wexog]
        n = [z.shape[0] for z in self._split_wexog]
        mn = np.asarray(mn)
        n = np.asarray(n)
        mnc = np.dot(mn.T, n[:, None] * mn) / n.sum()
        a, b = np.linalg.eigh(mnc)
        jj = np.argsort(-a)
        a = a[jj]
        b = b[:, jj]
        params = np.linalg.solve(self._covxr.T, b)
        results = DimReductionResults(self, params, eigs=a)
        return DimReductionResultsWrapper(results)

    def _regularized_objective(self, A):
        p = self.k_vars
        covx = self._covx
        mn = self._slice_means
        ph = self._slice_props
        v = 0
        A = np.reshape(A, (p, self.ndim))
        for k in range(self.ndim):
            u = np.dot(self.pen_mat, A[:, k])
            v += np.sum(u * u)
        covxa = np.dot(covx, A)
        q, _ = np.linalg.qr(covxa)
        qd = np.dot(q, np.dot(q.T, mn.T))
        qu = mn.T - qd
        v += np.dot(ph, (qu * qu).sum(0))
        return v

    def _regularized_grad(self, A):
        p = self.k_vars
        ndim = self.ndim
        covx = self._covx
        n_slice = self.n_slice
        mn = self._slice_means
        ph = self._slice_props
        A = A.reshape((p, ndim))
        gr = 2 * np.dot(self.pen_mat.T, np.dot(self.pen_mat, A))
        A = A.reshape((p, ndim))
        covxa = np.dot(covx, A)
        covx2a = np.dot(covx, covxa)
        Q = np.dot(covxa.T, covxa)
        Qi = np.linalg.inv(Q)
        jm = np.zeros((p, ndim))
        qcv = np.linalg.solve(Q, covxa.T)
        ft = [None] * (p * ndim)
        for q in range(p):
            for r in range(ndim):
                jm *= 0
                jm[q, r] = 1
                umat = np.dot(covx2a.T, jm)
                umat += umat.T
                umat = -np.dot(Qi, np.dot(umat, Qi))
                fmat = np.dot(np.dot(covx, jm), qcv)
                fmat += np.dot(covxa, np.dot(umat, covxa.T))
                fmat += np.dot(covxa, np.linalg.solve(Q, np.dot(jm.T, covx)))
                ft[q * ndim + r] = fmat
        ch = np.linalg.solve(Q, np.dot(covxa.T, mn.T))
        cu = mn - np.dot(covxa, ch).T
        for i in range(n_slice):
            u = cu[i, :]
            v = mn[i, :]
            for q in range(p):
                for r in range(ndim):
                    f = np.dot(u, np.dot(ft[q * ndim + r], v))
                    gr[q, r] -= 2 * ph[i] * f
        return gr.ravel()

    def fit_regularized(self, ndim=1, pen_mat=None, slice_n=20, maxiter=100, gtol=0.001, **kwargs):
        """
        Estimate the EDR space using regularized SIR.

        Parameters
        ----------
        ndim : int
            The number of EDR directions to estimate
        pen_mat : array_like
            A 2d array such that the squared Frobenius norm of
            `dot(pen_mat, dirs)`` is added to the objective function,
            where `dirs` is an orthogonal array whose columns span
            the estimated EDR space.
        slice_n : int, optional
            Target number of observations per slice
        maxiter :int
            The maximum number of iterations for estimating the EDR
            space.
        gtol : float
            If the norm of the gradient of the objective function
            falls below this value, the algorithm has converged.

        Returns
        -------
        A results class instance.

        Notes
        -----
        If each row of `exog` can be viewed as containing the values of a
        function evaluated at equally-spaced locations, then setting the
        rows of `pen_mat` to [[1, -2, 1, ...], [0, 1, -2, 1, ..], ...]
        will give smooth EDR coefficients.  This is a form of "functional
        SIR" using the squared second derivative as a penalty.

        References
        ----------
        L. Ferre, A.F. Yao (2003).  Functional sliced inverse regression
        analysis.  Statistics: a journal of theoretical and applied
        statistics 37(6) 475-488.
        """
        if len(kwargs) > 0:
            msg = 'SIR.fit_regularized does not take keyword arguments'
            warnings.warn(msg)
        if pen_mat is None:
            raise ValueError('pen_mat is a required argument')
        start_params = kwargs.get('start_params', None)
        slice_n = kwargs.get('slice_n', 20)
        n_slice = self.exog.shape[0] // slice_n
        ii = np.argsort(self.endog)
        x = self.exog[ii, :]
        x -= x.mean(0)
        covx = np.cov(x.T)
        split_exog = np.array_split(x, n_slice)
        mn = [z.mean(0) for z in split_exog]
        n = [z.shape[0] for z in split_exog]
        mn = np.asarray(mn)
        n = np.asarray(n)
        self._slice_props = n / n.sum()
        self.ndim = ndim
        self.k_vars = covx.shape[0]
        self.pen_mat = pen_mat
        self._covx = covx
        self.n_slice = n_slice
        self._slice_means = mn
        if start_params is None:
            params = np.zeros((self.k_vars, ndim))
            params[0:ndim, 0:ndim] = np.eye(ndim)
            params = params
        else:
            if start_params.shape[1] != ndim:
                msg = 'Shape of start_params is not compatible with ndim'
                raise ValueError(msg)
            params = start_params
        params, _, cnvrg = _grass_opt(params, self._regularized_objective, self._regularized_grad, maxiter, gtol)
        if not cnvrg:
            g = self._regularized_grad(params.ravel())
            gn = np.sqrt(np.dot(g, g))
            msg = 'SIR.fit_regularized did not converge, |g|=%f' % gn
            warnings.warn(msg)
        results = DimReductionResults(self, params, eigs=None)
        return DimReductionResultsWrapper(results)