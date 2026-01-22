from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
class LikelihoodModelResults(Results):
    """
    Class to contain results from likelihood models

    Parameters
    ----------
    model : LikelihoodModel instance or subclass instance
        LikelihoodModelResults holds a reference to the model that is fit.
    params : 1d array_like
        parameter estimates from estimated model
    normalized_cov_params : 2d array
       Normalized (before scaling) covariance of params. (dot(X.T,X))**-1
    scale : float
        For (some subset of models) scale will typically be the
        mean square error from the estimated model (sigma^2)

    Attributes
    ----------
    mle_retvals : dict
        Contains the values returned from the chosen optimization method if
        full_output is True during the fit.  Available only if the model
        is fit by maximum likelihood.  See notes below for the output from
        the different methods.
    mle_settings : dict
        Contains the arguments passed to the chosen optimization method.
        Available if the model is fit by maximum likelihood.  See
        LikelihoodModel.fit for more information.
    model : model instance
        LikelihoodResults contains a reference to the model that is fit.
    params : ndarray
        The parameters estimated for the model.
    scale : float
        The scaling factor of the model given during instantiation.
    tvalues : ndarray
        The t-values of the standard errors.


    Notes
    -----
    The covariance of params is given by scale times normalized_cov_params.

    Return values by solver if full_output is True during fit:

        'newton'
            fopt : float
                The value of the (negative) loglikelihood at its
                minimum.
            iterations : int
                Number of iterations performed.
            score : ndarray
                The score vector at the optimum.
            Hessian : ndarray
                The Hessian at the optimum.
            warnflag : int
                1 if maxiter is exceeded. 0 if successful convergence.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                List of solutions at each iteration.
        'nm'
            fopt : float
                The value of the (negative) loglikelihood at its
                minimum.
            iterations : int
                Number of iterations performed.
            warnflag : int
                1: Maximum number of function evaluations made.
                2: Maximum number of iterations reached.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                List of solutions at each iteration.
        'bfgs'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            gopt : float
                Value of gradient at minimum, which should be near 0.
            Hinv : ndarray
                value of the inverse Hessian matrix at minimum.  Note
                that this is just an approximation and will often be
                different from the value of the analytic Hessian.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            warnflag : int
                1: Maximum number of iterations exceeded. 2: Gradient
                and/or function calls are not changing.
            converged : bool
                True: converged.  False: did not converge.
            allvecs : list
                Results at each iteration.
        'lbfgs'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            gopt : float
                Value of gradient at minimum, which should be near 0.
            fcalls : int
                Number of calls to loglike.
            warnflag : int
                Warning flag:

                - 0 if converged
                - 1 if too many function evaluations or too many iterations
                - 2 if stopped for another reason

            converged : bool
                True: converged.  False: did not converge.
        'powell'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            direc : ndarray
                Current direction set.
            iterations : int
                Number of iterations performed.
            fcalls : int
                Number of calls to loglike.
            warnflag : int
                1: Maximum number of function evaluations. 2: Maximum number
                of iterations.
            converged : bool
                True : converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        'cg'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            warnflag : int
                1: Maximum number of iterations exceeded. 2: Gradient and/
                or function calls not changing.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        'ncg'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            hcalls : int
                Number of calls to hessian.
            warnflag : int
                1: Maximum number of iterations exceeded.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        """

    def __init__(self, model, params, normalized_cov_params=None, scale=1.0, **kwargs):
        super().__init__(model, params)
        self.normalized_cov_params = normalized_cov_params
        self.scale = scale
        self._use_t = False
        if 'use_t' in kwargs:
            use_t = kwargs['use_t']
            self.use_t = use_t if use_t is not None else False
        if 'cov_type' in kwargs:
            cov_type = kwargs.get('cov_type', 'nonrobust')
            cov_kwds = kwargs.get('cov_kwds', {})
            if cov_type == 'nonrobust':
                self.cov_type = 'nonrobust'
                self.cov_kwds = {'description': 'Standard Errors assume that the ' + 'covariance matrix of the errors is correctly ' + 'specified.'}
            else:
                from statsmodels.base.covtype import get_robustcov_results
                if cov_kwds is None:
                    cov_kwds = {}
                use_t = self.use_t
                get_robustcov_results(self, cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)

    def normalized_cov_params(self):
        """See specific model class docstring"""
        raise NotImplementedError

    def _get_robustcov_results(self, cov_type='nonrobust', use_self=True, use_t=None, **cov_kwds):
        if use_self is False:
            raise ValueError('use_self should have been removed long ago.  See GH#4401')
        from statsmodels.base.covtype import get_robustcov_results
        if cov_kwds is None:
            cov_kwds = {}
        if cov_type == 'nonrobust':
            self.cov_type = 'nonrobust'
            self.cov_kwds = {'description': 'Standard Errors assume that the ' + 'covariance matrix of the errors is correctly ' + 'specified.'}
        else:
            get_robustcov_results(self, cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)

    @property
    def use_t(self):
        """Flag indicating to use the Student's distribution in inference."""
        return self._use_t

    @use_t.setter
    def use_t(self, value):
        self._use_t = bool(value)

    @cached_value
    def llf(self):
        """Log-likelihood of model"""
        return self.model.loglike(self.params)

    @cached_value
    def bse(self):
        """The standard errors of the parameter estimates."""
        if not hasattr(self, 'cov_params_default') and self.normalized_cov_params is None:
            bse_ = np.empty(len(self.params))
            bse_[:] = np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                bse_ = np.sqrt(np.diag(self.cov_params()))
        return bse_

    @cached_value
    def tvalues(self):
        """
        Return the t-statistic for a given parameter estimate.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return self.params / self.bse

    @cached_value
    def pvalues(self):
        """The two-tailed p values for the t-stats of the params."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if self.use_t:
                df_resid = getattr(self, 'df_resid_inference', self.df_resid)
                return stats.t.sf(np.abs(self.tvalues), df_resid) * 2
            else:
                return stats.norm.sf(np.abs(self.tvalues)) * 2

    def cov_params(self, r_matrix=None, column=None, scale=None, cov_p=None, other=None):
        """
        Compute the variance/covariance matrix.

        The variance/covariance matrix can be of a linear contrast of the
        estimated parameters or all params multiplied by scale which will
        usually be an estimate of sigma^2.  Scale is assumed to be a scalar.

        Parameters
        ----------
        r_matrix : array_like
            Can be 1d, or 2d.  Can be used alone or with other.
        column : array_like, optional
            Must be used on its own.  Can be 0d or 1d see below.
        scale : float, optional
            Can be specified or not.  Default is None, which means that
            the scale argument is taken from the model.
        cov_p : ndarray, optional
            The covariance of the parameters. If not provided, this value is
            read from `self.normalized_cov_params` or
            `self.cov_params_default`.
        other : array_like, optional
            Can be used when r_matrix is specified.

        Returns
        -------
        ndarray
            The covariance matrix of the parameter estimates or of linear
            combination of parameter estimates. See Notes.

        Notes
        -----
        (The below are assumed to be in matrix notation.)

        If no argument is specified returns the covariance matrix of a model
        ``(scale)*(X.T X)^(-1)``

        If contrast is specified it pre and post-multiplies as follows
        ``(scale) * r_matrix (X.T X)^(-1) r_matrix.T``

        If contrast and other are specified returns
        ``(scale) * r_matrix (X.T X)^(-1) other.T``

        If column is specified returns
        ``(scale) * (X.T X)^(-1)[column,column]`` if column is 0d

        OR

        ``(scale) * (X.T X)^(-1)[column][:,column]`` if column is 1d
        """
        if hasattr(self, 'mle_settings') and self.mle_settings['optimizer'] in ['l1', 'l1_cvxopt_cp']:
            dot_fun = nan_dot
        else:
            dot_fun = np.dot
        if cov_p is None and self.normalized_cov_params is None and (not hasattr(self, 'cov_params_default')):
            raise ValueError('need covariance of parameters for computing (unnormalized) covariances')
        if column is not None and (r_matrix is not None or other is not None):
            raise ValueError('Column should be specified without other arguments.')
        if other is not None and r_matrix is None:
            raise ValueError('other can only be specified with r_matrix')
        if cov_p is None:
            if hasattr(self, 'cov_params_default'):
                cov_p = self.cov_params_default
            else:
                if scale is None:
                    scale = self.scale
                cov_p = self.normalized_cov_params * scale
        if column is not None:
            column = np.asarray(column)
            if column.shape == ():
                return cov_p[column, column]
            else:
                return cov_p[column[:, None], column]
        elif r_matrix is not None:
            r_matrix = np.asarray(r_matrix)
            if r_matrix.shape == ():
                raise ValueError('r_matrix should be 1d or 2d')
            if other is None:
                other = r_matrix
            else:
                other = np.asarray(other)
            tmp = dot_fun(r_matrix, dot_fun(cov_p, np.transpose(other)))
            return tmp
        else:
            return cov_p

    def t_test(self, r_matrix, cov_p=None, use_t=None):
        """
        Compute a t-test for a each linear hypothesis of the form Rb = q.

        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:

            - array : If an array is given, a p x k 2d array or length k 1d
              array specifying the linear restrictions. It is assumed
              that the linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q). If q is given,
              can be either a scalar or a length p row vector.

        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        use_t : bool, optional
            If use_t is None, then the default of the model is used. If use_t
            is True, then the p-values are based on the t distribution. If
            use_t is False, then the p-values are based on the normal
            distribution.

        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.
            The available results have the same elements as the parameter table
            in `summary()`.

        See Also
        --------
        tvalues : Individual t statistics for the estimated parameters.
        f_test : Perform an F tests on model parameters.
        patsy.DesignInfo.linear_constraint : Specify a linear constraint.

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> r = np.zeros_like(results.params)
        >>> r[5:] = [1,-1]
        >>> print(r)
        [ 0.  0.  0.  0.  0.  1. -1.]

        r tests that the coefficients on the 5th and 6th independent
        variable are the same.

        >>> T_test = results.t_test(r)
        >>> print(T_test)
                                     Test for Constraints
        ==============================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
        ------------------------------------------------------------------------------
        c0         -1829.2026    455.391     -4.017      0.003   -2859.368    -799.037
        ==============================================================================
        >>> T_test.effect
        -1829.2025687192481
        >>> T_test.sd
        455.39079425193762
        >>> T_test.tvalue
        -4.0167754636411717
        >>> T_test.pvalue
        0.0015163772380899498

        Alternatively, you can specify the hypothesis tests using a string

        >>> from statsmodels.formula.api import ols
        >>> dta = sm.datasets.longley.load_pandas().data
        >>> formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
        >>> results = ols(formula, dta).fit()
        >>> hypotheses = 'GNPDEFL = GNP, UNEMP = 2, YEAR/1829 = 1'
        >>> t_test = results.t_test(hypotheses)
        >>> print(t_test)
                                     Test for Constraints
        ==============================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
        ------------------------------------------------------------------------------
        c0            15.0977     84.937      0.178      0.863    -177.042     207.238
        c1            -2.0202      0.488     -8.231      0.000      -3.125      -0.915
        c2             1.0001      0.249      0.000      1.000       0.437       1.563
        ==============================================================================
        """
        from patsy import DesignInfo
        use_t = bool_like(use_t, 'use_t', strict=True, optional=True)
        if self.params.ndim == 2:
            names = [f'y{i[0]}_{i[1]}' for i in self.model.data.cov_names]
        else:
            names = self.model.data.cov_names
        LC = DesignInfo(names).linear_constraint(r_matrix)
        r_matrix, q_matrix = (LC.coefs, LC.constants)
        num_ttests = r_matrix.shape[0]
        num_params = r_matrix.shape[1]
        if cov_p is None and self.normalized_cov_params is None and (not hasattr(self, 'cov_params_default')):
            raise ValueError('Need covariance of parameters for computing T statistics')
        params = self.params.ravel(order='F')
        if num_params != params.shape[0]:
            raise ValueError('r_matrix and params are not aligned')
        if q_matrix is None:
            q_matrix = np.zeros(num_ttests)
        else:
            q_matrix = np.asarray(q_matrix)
            q_matrix = q_matrix.squeeze()
        if q_matrix.size > 1:
            if q_matrix.shape[0] != num_ttests:
                raise ValueError('r_matrix and q_matrix must have the same number of rows')
        if use_t is None:
            use_t = hasattr(self, 'use_t') and self.use_t
        _effect = np.dot(r_matrix, params)
        if num_ttests > 1:
            _sd = np.sqrt(np.diag(self.cov_params(r_matrix=r_matrix, cov_p=cov_p)))
        else:
            _sd = np.sqrt(self.cov_params(r_matrix=r_matrix, cov_p=cov_p))
        _t = (_effect - q_matrix) * recipr(_sd)
        df_resid = getattr(self, 'df_resid_inference', self.df_resid)
        if use_t:
            return ContrastResults(effect=_effect, t=_t, sd=_sd, df_denom=df_resid)
        else:
            return ContrastResults(effect=_effect, statistic=_t, sd=_sd, df_denom=df_resid, distribution='norm')

    def f_test(self, r_matrix, cov_p=None, invcov=None):
        """
        Compute the F-test for a joint linear hypothesis.

        This is a special case of `wald_test` that always uses the F
        distribution.

        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:

            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed
              that the linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length k row vector.

        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        invcov : array_like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.

        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.

        See Also
        --------
        t_test : Perform a single hypothesis test.
        wald_test : Perform a Wald-test using a quadratic form.
        statsmodels.stats.contrast.ContrastResults : Test results.
        patsy.DesignInfo.linear_constraint : Specify a linear constraint.

        Notes
        -----
        The matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> A = np.identity(len(results.params))
        >>> A = A[1:,:]

        This tests that each coefficient is jointly statistically
        significantly different from zero.

        >>> print(results.f_test(A))
        <F test: F=array([[ 330.28533923]]), p=4.984030528700946e-10, df_denom=9, df_num=6>

        Compare this to

        >>> results.fvalue
        330.2853392346658
        >>> results.f_pvalue
        4.98403096572e-10

        >>> B = np.array(([0,0,1,-1,0,0,0],[0,0,0,0,0,1,-1]))

        This tests that the coefficient on the 2nd and 3rd regressors are
        equal and jointly that the coefficient on the 5th and 6th regressors
        are equal.

        >>> print(results.f_test(B))
        <F test: F=array([[ 9.74046187]]), p=0.005605288531708235, df_denom=9, df_num=2>

        Alternatively, you can specify the hypothesis tests using a string

        >>> from statsmodels.datasets import longley
        >>> from statsmodels.formula.api import ols
        >>> dta = longley.load_pandas().data
        >>> formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
        >>> results = ols(formula, dta).fit()
        >>> hypotheses = '(GNPDEFL = GNP), (UNEMP = 2), (YEAR/1829 = 1)'
        >>> f_test = results.f_test(hypotheses)
        >>> print(f_test)
        <F test: F=array([[ 144.17976065]]), p=6.322026217355609e-08, df_denom=9, df_num=3>
        """
        res = self.wald_test(r_matrix, cov_p=cov_p, invcov=invcov, use_f=True, scalar=True)
        return res

    def wald_test(self, r_matrix, cov_p=None, invcov=None, use_f=None, df_constraints=None, scalar=None):
        """
        Compute a Wald-test for a joint linear hypothesis.

        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:

            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed that the
              linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length p row vector.

        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        invcov : array_like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.
        use_f : bool
            If True, then the F-distribution is used. If False, then the
            asymptotic distribution, chisquare is used. If use_f is None, then
            the F distribution is used if the model specifies that use_t is True.
            The test statistic is proportionally adjusted for the distribution
            by the number of constraints in the hypothesis.
        df_constraints : int, optional
            The number of constraints. If not provided the number of
            constraints is determined from r_matrix.
        scalar : bool, optional
            Flag indicating whether the Wald test statistic should be returned
            as a sclar float. The current behavior is to return an array.
            This will switch to a scalar float after 0.14 is released. To
            get the future behavior now, set scalar to True. To silence
            the warning and retain the legacy behavior, set scalar to
            False.

        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.

        See Also
        --------
        f_test : Perform an F tests on model parameters.
        t_test : Perform a single hypothesis test.
        statsmodels.stats.contrast.ContrastResults : Test results.
        patsy.DesignInfo.linear_constraint : Specify a linear constraint.

        Notes
        -----
        The matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.
        """
        use_f = bool_like(use_f, 'use_f', strict=True, optional=True)
        scalar = bool_like(scalar, 'scalar', strict=True, optional=True)
        if use_f is None:
            use_f = hasattr(self, 'use_t') and self.use_t
        from patsy import DesignInfo
        if self.params.ndim == 2:
            names = [f'y{i[0]}_{i[1]}' for i in self.model.data.cov_names]
        else:
            names = self.model.data.cov_names
        params = self.params.ravel(order='F')
        LC = DesignInfo(names).linear_constraint(r_matrix)
        r_matrix, q_matrix = (LC.coefs, LC.constants)
        if self.normalized_cov_params is None and cov_p is None and (invcov is None) and (not hasattr(self, 'cov_params_default')):
            raise ValueError('need covariance of parameters for computing F statistics')
        cparams = np.dot(r_matrix, params[:, None])
        J = float(r_matrix.shape[0])
        if q_matrix is None:
            q_matrix = np.zeros(J)
        else:
            q_matrix = np.asarray(q_matrix)
        if q_matrix.ndim == 1:
            q_matrix = q_matrix[:, None]
            if q_matrix.shape[0] != J:
                raise ValueError('r_matrix and q_matrix must have the same number of rows')
        Rbq = cparams - q_matrix
        if invcov is None:
            cov_p = self.cov_params(r_matrix=r_matrix, cov_p=cov_p)
            if np.isnan(cov_p).max():
                raise ValueError('r_matrix performs f_test for using dimensions that are asymptotically non-normal')
            invcov = np.linalg.pinv(cov_p)
            J_ = np.linalg.matrix_rank(cov_p)
            if J_ < J:
                warnings.warn('covariance of constraints does not have full rank. The number of constraints is %d, but rank is %d' % (J, J_), ValueWarning)
                J = J_
        if df_constraints is not None:
            J = df_constraints
        if hasattr(self, 'mle_settings') and self.mle_settings['optimizer'] in ['l1', 'l1_cvxopt_cp']:
            F = nan_dot(nan_dot(Rbq.T, invcov), Rbq)
        else:
            F = np.dot(np.dot(Rbq.T, invcov), Rbq)
        df_resid = getattr(self, 'df_resid_inference', self.df_resid)
        if scalar is None:
            warnings.warn('The behavior of wald_test will change after 0.14 to returning scalar test statistic values. To get the future behavior now, set scalar to True. To silence this message while retaining the legacy behavior, set scalar to False.', FutureWarning)
            scalar = False
        if scalar and F.size == 1:
            F = float(np.squeeze(F))
        if use_f:
            F /= J
            return ContrastResults(F=F, df_denom=df_resid, df_num=J)
        else:
            return ContrastResults(chi2=F, df_denom=J, statistic=F, distribution='chi2', distargs=(J,))

    def wald_test_terms(self, skip_single=False, extra_constraints=None, combine_terms=None, scalar=None):
        """
        Compute a sequence of Wald tests for terms over multiple columns.

        This computes joined Wald tests for the hypothesis that all
        coefficients corresponding to a `term` are zero.
        `Terms` are defined by the underlying formula or by string matching.

        Parameters
        ----------
        skip_single : bool
            If true, then terms that consist only of a single column and,
            therefore, refers only to a single parameter is skipped.
            If false, then all terms are included.
        extra_constraints : ndarray
            Additional constraints to test. Note that this input has not been
            tested.
        combine_terms : {list[str], None}
            Each string in this list is matched to the name of the terms or
            the name of the exogenous variables. All columns whose name
            includes that string are combined in one joint test.
        scalar : bool, optional
            Flag indicating whether the Wald test statistic should be returned
            as a sclar float. The current behavior is to return an array.
            This will switch to a scalar float after 0.14 is released. To
            get the future behavior now, set scalar to True. To silence
            the warning and retain the legacy behavior, set scalar to
            False.

        Returns
        -------
        WaldTestResults
            The result instance contains `table` which is a pandas DataFrame
            with the test results: test statistic, degrees of freedom and
            pvalues.

        Examples
        --------
        >>> res_ols = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()
        >>> res_ols.wald_test_terms()
        <class 'statsmodels.stats.contrast.WaldTestResults'>
                                                  F                P>F  df constraint  df denom
        Intercept                        279.754525  2.37985521351e-22              1        51
        C(Duration, Sum)                   5.367071    0.0245738436636              1        51
        C(Weight, Sum)                    12.432445  3.99943118767e-05              2        51
        C(Duration, Sum):C(Weight, Sum)    0.176002      0.83912310946              2        51

        >>> res_poi = Poisson.from_formula("Days ~ C(Weight) * C(Duration)",                                            data).fit(cov_type='HC0')
        >>> wt = res_poi.wald_test_terms(skip_single=False,                                          combine_terms=['Duration', 'Weight'])
        >>> print(wt)
                                    chi2             P>chi2  df constraint
        Intercept              15.695625  7.43960374424e-05              1
        C(Weight)              16.132616  0.000313940174705              2
        C(Duration)             1.009147     0.315107378931              1
        C(Weight):C(Duration)   0.216694     0.897315972824              2
        Duration               11.187849     0.010752286833              3
        Weight                 30.263368  4.32586407145e-06              4
        """
        from collections import defaultdict
        result = self
        if extra_constraints is None:
            extra_constraints = []
        if combine_terms is None:
            combine_terms = []
        design_info = getattr(result.model.data, 'design_info', None)
        if design_info is None and extra_constraints is None:
            raise ValueError('no constraints, nothing to do')
        identity = np.eye(len(result.params))
        constraints = []
        combined = defaultdict(list)
        if design_info is not None:
            for term in design_info.terms:
                cols = design_info.slice(term)
                name = term.name()
                constraint_matrix = identity[cols]
                for cname in combine_terms:
                    if cname in name:
                        combined[cname].append(constraint_matrix)
                k_constraint = constraint_matrix.shape[0]
                if skip_single:
                    if k_constraint == 1:
                        continue
                constraints.append((name, constraint_matrix))
            combined_constraints = []
            for cname in combine_terms:
                combined_constraints.append((cname, np.vstack(combined[cname])))
        else:
            for col, name in enumerate(result.model.exog_names):
                constraint_matrix = np.atleast_2d(identity[col])
                for cname in combine_terms:
                    if cname in name:
                        combined[cname].append(constraint_matrix)
                if skip_single:
                    continue
                constraints.append((name, constraint_matrix))
            combined_constraints = []
            for cname in combine_terms:
                combined_constraints.append((cname, np.vstack(combined[cname])))
        use_t = result.use_t
        distribution = ['chi2', 'F'][use_t]
        res_wald = []
        index = []
        for name, constraint in constraints + combined_constraints + extra_constraints:
            wt = result.wald_test(constraint, scalar=scalar)
            row = [wt.statistic, wt.pvalue, constraint.shape[0]]
            if use_t:
                row.append(wt.df_denom)
            res_wald.append(row)
            index.append(name)
        col_names = ['statistic', 'pvalue', 'df_constraint']
        if use_t:
            col_names.append('df_denom')
        from pandas import DataFrame
        table = DataFrame(res_wald, index=index, columns=col_names)
        res = WaldTestResults(None, distribution, None, table=table)
        res.temp = constraints + combined_constraints + extra_constraints
        return res

    def t_test_pairwise(self, term_name, method='hs', alpha=0.05, factor_labels=None):
        """
        Perform pairwise t_test with multiple testing corrected p-values.

        This uses the formula design_info encoding contrast matrix and should
        work for all encodings of a main effect.

        Parameters
        ----------
        term_name : str
            The name of the term for which pairwise comparisons are computed.
            Term names for categorical effects are created by patsy and
            correspond to the main part of the exog names.
        method : {str, list[str]}
            The multiple testing p-value correction to apply. The default is
            'hs'. See stats.multipletesting.
        alpha : float
            The significance level for multiple testing reject decision.
        factor_labels : {list[str], None}
            Labels for the factor levels used for pairwise labels. If not
            provided, then the labels from the formula design_info are used.

        Returns
        -------
        MultiCompResult
            The results are stored as attributes, the main attributes are the
            following two. Other attributes are added for debugging purposes
            or as background information.

            - result_frame : pandas DataFrame with t_test results and multiple
              testing corrected p-values.
            - contrasts : matrix of constraints of the null hypothesis in the
              t_test.

        Notes
        -----
        Status: experimental. Currently only checked for treatment coding with
        and without specified reference level.

        Currently there are no multiple testing corrected confidence intervals
        available.

        Examples
        --------
        >>> res = ols("np.log(Days+1) ~ C(Weight) + C(Duration)", data).fit()
        >>> pw = res.t_test_pairwise("C(Weight)")
        >>> pw.result_frame
                 coef   std err         t         P>|t|  Conf. Int. Low
        2-1  0.632315  0.230003  2.749157  8.028083e-03        0.171563
        3-1  1.302555  0.230003  5.663201  5.331513e-07        0.841803
        3-2  0.670240  0.230003  2.914044  5.119126e-03        0.209488
             Conf. Int. Upp.  pvalue-hs reject-hs
        2-1         1.093067   0.010212      True
        3-1         1.763307   0.000002      True
        3-2         1.130992   0.010212      True
        """
        res = t_test_pairwise(self, term_name, method=method, alpha=alpha, factor_labels=factor_labels)
        return res

    def _get_wald_nonlinear(self, func, deriv=None):
        """Experimental method for nonlinear prediction and tests

        Parameters
        ----------
        func : callable, f(params)
            nonlinear function of the estimation parameters. The return of
            the function can be vector valued, i.e. a 1-D array
        deriv : function or None
            first derivative or Jacobian of func. If deriv is None, then a
            numerical derivative will be used. If func returns a 1-D array,
            then the `deriv` should have rows corresponding to the elements
            of the return of func.

        Returns
        -------
        nl : instance of `NonlinearDeltaCov` with attributes and methods to
            calculate the results for the prediction or tests

        """
        from statsmodels.stats._delta_method import NonlinearDeltaCov
        func_args = None
        nl = NonlinearDeltaCov(func, self.params, self.cov_params(), deriv=deriv, func_args=func_args)
        return nl

    def conf_int(self, alpha=0.05, cols=None):
        """
        Construct confidence interval for the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval. The default
            `alpha` = .05 returns a 95% confidence interval.
        cols : array_like, optional
            Specifies which confidence intervals to return.

        .. deprecated: 0.13

           cols is deprecated and will be removed after 0.14 is released.
           cols only works when inputs are NumPy arrays and will fail
           when using pandas Series or DataFrames as input. You can
           subset the confidence intervals using slices.

        Returns
        -------
        array_like
            Each row contains [lower, upper] limits of the confidence interval
            for the corresponding parameter. The first column contains all
            lower, the second column contains all upper limits.

        Notes
        -----
        The confidence interval is based on the standard normal distribution
        if self.use_t is False. If self.use_t is True, then uses a Student's t
        with self.df_resid_inference (or self.df_resid if df_resid_inference is
        not defined) degrees of freedom.

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> results.conf_int()
        array([[-5496529.48322745, -1467987.78596704],
               [    -177.02903529,      207.15277984],
               [      -0.1115811 ,        0.03994274],
               [      -3.12506664,       -0.91539297],
               [      -1.5179487 ,       -0.54850503],
               [      -0.56251721,        0.460309  ],
               [     798.7875153 ,     2859.51541392]])

        >>> results.conf_int(cols=(2,3))
        array([[-0.1115811 ,  0.03994274],
               [-3.12506664, -0.91539297]])
        """
        bse = self.bse
        if self.use_t:
            dist = stats.t
            df_resid = getattr(self, 'df_resid_inference', self.df_resid)
            q = dist.ppf(1 - alpha / 2, df_resid)
        else:
            dist = stats.norm
            q = dist.ppf(1 - alpha / 2)
        params = self.params
        lower = params - q * bse
        upper = params + q * bse
        if cols is not None:
            warnings.warn('cols is deprecated and will be removed after 0.14 is released. cols only works when inputs are NumPy arrays and will fail when using pandas Series or DataFrames as input. Subsets of confidence intervals can be selected using slices of the full confidence interval array.', FutureWarning)
            cols = np.asarray(cols)
            lower = lower[cols]
            upper = upper[cols]
        return np.asarray(lzip(lower, upper))

    def save(self, fname, remove_data=False):
        """
        Save a pickle of this instance.

        Parameters
        ----------
        fname : {str, handle}
            A string filename or a file handle.
        remove_data : bool
            If False (default), then the instance is pickled without changes.
            If True, then all arrays with length nobs are set to None before
            pickling. See the remove_data method.
            In some cases not all arrays will be set to None.

        Notes
        -----
        If remove_data is true and the model result does not implement a
        remove_data method then this will raise an exception.
        """
        from statsmodels.iolib.smpickle import save_pickle
        if remove_data:
            self.remove_data()
        save_pickle(self, fname)

    @classmethod
    def load(cls, fname):
        """
        Load a pickled results instance

        .. warning::

           Loading pickled models is not secure against erroneous or
           maliciously constructed data. Never unpickle data received from
           an untrusted or unauthenticated source.

        Parameters
        ----------
        fname : {str, handle, pathlib.Path}
            A string filename or a file handle.

        Returns
        -------
        Results
            The unpickled results instance.
        """
        from statsmodels.iolib.smpickle import load_pickle
        return load_pickle(fname)

    def remove_data(self):
        """
        Remove data arrays, all nobs arrays from result and model.

        This reduces the size of the instance, so it can be pickled with less
        memory. Currently tested for use with predict from an unpickled
        results and model instance.

        .. warning::

           Since data and some intermediate results have been removed
           calculating new statistics that require them will raise exceptions.
           The exception will occur the first time an attribute is accessed
           that has been set to None.

        Not fully tested for time series models, tsa, and might delete too much
        for prediction or not all that would be possible.

        The lists of arrays to delete are maintained as attributes of
        the result and model instance, except for cached values. These
        lists could be changed before calling remove_data.

        The attributes to remove are named in:

        model._data_attr : arrays attached to both the model instance
            and the results instance with the same attribute name.

        result._data_in_cache : arrays that may exist as values in
            result._cache

        result._data_attr_model : arrays attached to the model
            instance but not to the results instance
        """
        cls = self.__class__
        cls_attrs = {}
        for name in dir(cls):
            try:
                attr = object.__getattribute__(cls, name)
            except AttributeError:
                pass
            else:
                cls_attrs[name] = attr
        data_attrs = [x for x in cls_attrs if isinstance(cls_attrs[x], cached_data)]
        for name in data_attrs:
            self._cache[name] = None

        def wipe(obj, att):
            p = att.split('.')
            att_ = p.pop(-1)
            try:
                obj_ = reduce(getattr, [obj] + p)
                if hasattr(obj_, att_):
                    setattr(obj_, att_, None)
            except AttributeError:
                pass
        model_only = ['model.' + i for i in getattr(self, '_data_attr_model', [])]
        model_attr = ['model.' + i for i in self.model._data_attr]
        for att in self._data_attr + model_attr + model_only:
            if att in data_attrs:
                continue
            wipe(self, att)
        for key in self._data_in_cache:
            try:
                self._cache[key] = None
            except (AttributeError, KeyError):
                pass