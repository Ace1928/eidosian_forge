from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
class GEEResults(GLMResults):
    __doc__ = 'This class summarizes the fit of a marginal regression model using GEE.\n' + _gee_results_doc

    def __init__(self, model, params, cov_params, scale, cov_type='robust', use_t=False, regularized=False, **kwds):
        super().__init__(model, params, normalized_cov_params=cov_params, scale=scale)
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self.family = model.family
        attr_kwds = kwds.pop('attr_kwds', {})
        self.__dict__.update(attr_kwds)
        if not (hasattr(self, 'cov_type') and hasattr(self, 'cov_params_default')):
            self.cov_type = cov_type
            covariance_type = self.cov_type.lower()
            allowed_covariances = ['robust', 'naive', 'bias_reduced']
            if covariance_type not in allowed_covariances:
                msg = 'GEE: `cov_type` must be one of ' + ', '.join(allowed_covariances)
                raise ValueError(msg)
            if cov_type == 'robust':
                cov = self.cov_robust
            elif cov_type == 'naive':
                cov = self.cov_naive
            elif cov_type == 'bias_reduced':
                cov = self.cov_robust_bc
            self.cov_params_default = cov
        elif self.cov_type != cov_type:
            raise ValueError('cov_type in argument is different from already attached cov_type')

    @cache_readonly
    def resid(self):
        """
        The response residuals.
        """
        return self.resid_response

    def standard_errors(self, cov_type='robust'):
        """
        This is a convenience function that returns the standard
        errors for any covariance type.  The value of `bse` is the
        standard errors for whichever covariance type is specified as
        an argument to `fit` (defaults to "robust").

        Parameters
        ----------
        cov_type : str
            One of "robust", "naive", or "bias_reduced".  Determines
            the covariance used to compute standard errors.  Defaults
            to "robust".
        """
        covariance_type = cov_type.lower()
        allowed_covariances = ['robust', 'naive', 'bias_reduced']
        if covariance_type not in allowed_covariances:
            msg = 'GEE: `covariance_type` must be one of ' + ', '.join(allowed_covariances)
            raise ValueError(msg)
        if covariance_type == 'robust':
            return np.sqrt(np.diag(self.cov_robust))
        elif covariance_type == 'naive':
            return np.sqrt(np.diag(self.cov_naive))
        elif covariance_type == 'bias_reduced':
            if self.cov_robust_bc is None:
                raise ValueError('GEE: `bias_reduced` covariance not available')
            return np.sqrt(np.diag(self.cov_robust_bc))

    @cache_readonly
    def bse(self):
        return self.standard_errors(self.cov_type)

    def score_test(self):
        """
        Return the results of a score test for a linear constraint.

        Returns
        -------
        A\x7fdictionary containing the p-value, the test statistic,
        and the degrees of freedom for the score test.

        Notes
        -----
        See also GEE.compare_score_test for an alternative way to perform
        a score test.  GEEResults.score_test is more general, in that it
        supports testing arbitrary linear equality constraints.   However
        GEE.compare_score_test might be easier to use when comparing
        two explicit models.

        References
        ----------
        Xu Guo and Wei Pan (2002). "Small sample performance of the score
        test in GEE".
        http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf
        """
        if not hasattr(self.model, 'score_test_results'):
            msg = 'score_test on results instance only available when '
            msg += ' model was fit with constraints'
            raise ValueError(msg)
        return self.model.score_test_results

    @cache_readonly
    def resid_split(self):
        """
        Returns the residuals, the endogeneous data minus the fitted
        values from the model.  The residuals are returned as a list
        of arrays containing the residuals for each cluster.
        """
        sresid = []
        for v in self.model.group_labels:
            ii = self.model.group_indices[v]
            sresid.append(self.resid[ii])
        return sresid

    @cache_readonly
    def resid_centered(self):
        """
        Returns the residuals centered within each group.
        """
        cresid = self.resid.copy()
        for v in self.model.group_labels:
            ii = self.model.group_indices[v]
            cresid[ii] -= cresid[ii].mean()
        return cresid

    @cache_readonly
    def resid_centered_split(self):
        """
        Returns the residuals centered within each group.  The
        residuals are returned as a list of arrays containing the
        centered residuals for each cluster.
        """
        sresid = []
        for v in self.model.group_labels:
            ii = self.model.group_indices[v]
            sresid.append(self.centered_resid[ii])
        return sresid

    def qic(self, scale=None, n_step=1000):
        """
        Returns the QIC and QICu information criteria.

        See GEE.qic for documentation.
        """
        if scale is None:
            warnings.warn('QIC values obtained using scale=None are not appropriate for comparing models')
        if scale is None:
            scale = self.scale
        _, qic, qicu = self.model.qic(self.params, scale, self.cov_params(), n_step=n_step)
        return (qic, qicu)
    split_resid = resid_split
    centered_resid = resid_centered
    split_centered_resid = resid_centered_split

    @Appender(_plot_added_variable_doc % {'extra_params_doc': ''})
    def plot_added_variable(self, focus_exog, resid_type=None, use_glm_weights=True, fit_kwargs=None, ax=None):
        from statsmodels.graphics.regressionplots import plot_added_variable
        fig = plot_added_variable(self, focus_exog, resid_type=resid_type, use_glm_weights=use_glm_weights, fit_kwargs=fit_kwargs, ax=ax)
        return fig

    @Appender(_plot_partial_residuals_doc % {'extra_params_doc': ''})
    def plot_partial_residuals(self, focus_exog, ax=None):
        from statsmodels.graphics.regressionplots import plot_partial_residuals
        return plot_partial_residuals(self, focus_exog, ax=ax)

    @Appender(_plot_ceres_residuals_doc % {'extra_params_doc': ''})
    def plot_ceres_residuals(self, focus_exog, frac=0.66, cond_means=None, ax=None):
        from statsmodels.graphics.regressionplots import plot_ceres_residuals
        return plot_ceres_residuals(self, focus_exog, frac, cond_means=cond_means, ax=ax)

    def conf_int(self, alpha=0.05, cols=None, cov_type=None):
        """
        Returns confidence intervals for the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
             The `alpha` level for the confidence interval.  i.e., The
             default `alpha` = .05 returns a 95% confidence interval.
        cols : array_like, optional
             `cols` specifies which confidence intervals to return
        cov_type : str
             The covariance type used for computing standard errors;
             must be one of 'robust', 'naive', and 'bias reduced'.
             See `GEE` for details.

        Notes
        -----
        The confidence interval is based on the Gaussian distribution.
        """
        if cov_type is None:
            bse = self.bse
        else:
            bse = self.standard_errors(cov_type=cov_type)
        params = self.params
        dist = stats.norm
        q = dist.ppf(1 - alpha / 2)
        if cols is None:
            lower = self.params - q * bse
            upper = self.params + q * bse
        else:
            cols = np.asarray(cols)
            lower = params[cols] - q * bse[cols]
            upper = params[cols] + q * bse[cols]
        return np.asarray(lzip(lower, upper))

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        Summarize the GEE regression results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is `var_#` for ## in
            the number of regressors. Must match the number of parameters in
            the model
        title : str, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals
        cov_type : str
            The covariance type used to compute the standard errors;
            one of 'robust' (the usual robust sandwich-type covariance
            estimate), 'naive' (ignores dependence), and 'bias
            reduced' (the Mancl/DeRouen estimate).

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary results
        """
        top_left = [('Dep. Variable:', None), ('Model:', None), ('Method:', ['Generalized']), ('', ['Estimating Equations']), ('Family:', [self.model.family.__class__.__name__]), ('Dependence structure:', [self.model.cov_struct.__class__.__name__]), ('Date:', None), ('Covariance type: ', [self.cov_type])]
        NY = [len(y) for y in self.model.endog_li]
        top_right = [('No. Observations:', [sum(NY)]), ('No. clusters:', [len(self.model.endog_li)]), ('Min. cluster size:', [min(NY)]), ('Max. cluster size:', [max(NY)]), ('Mean cluster size:', ['%.1f' % np.mean(NY)]), ('Num. iterations:', ['%d' % len(self.fit_history['params'])]), ('Scale:', ['%.3f' % self.scale]), ('Time:', None)]
        skew1 = stats.skew(self.resid)
        kurt1 = stats.kurtosis(self.resid)
        skew2 = stats.skew(self.centered_resid)
        kurt2 = stats.kurtosis(self.centered_resid)
        diagn_left = [('Skew:', ['%12.4f' % skew1]), ('Centered skew:', ['%12.4f' % skew2])]
        diagn_right = [('Kurtosis:', ['%12.4f' % kurt1]), ('Centered kurtosis:', ['%12.4f' % kurt2])]
        if title is None:
            title = self.model.__class__.__name__ + ' ' + 'Regression Results'
        if xname is None:
            xname = self.model.exog_names
        if yname is None:
            yname = self.model.endog_names
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=False)
        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right, yname=yname, xname=xname, title='')
        return smry

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation. If `at` is 'all'
              only margeff will be available.

            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            Options are:

            - 'dydx' - dy/dx - No transformation is made and marginal effects
              are returned.  This is the default.
            - 'eyex' - estimate elasticities of variables in `exog` --
              d(lny)/d(lnx)
            - 'dyex' - estimate semi-elasticity -- dy/d(lnx)
            - 'eydx' - estimate semi-elasticity -- d(lny)/dx

            Note that tranformations are done after each observation is
            calculated.  Semi-elasticities for binary variables are computed
            using the midpoint method. 'dyex' and 'eyex' do not make sense
            for discrete variables.
        atexog : array_like, optional
            Optionally, you can provide the exogenous variables over which to
            get the marginal effects.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        dummy : bool, optional
            If False, treats binary variables (if present) as continuous.  This
            is the default.  Else if True, treats binary variables as
            changing from 0 to 1.  Note that any variable that is either 0 or 1
            is treated as binary.  Each binary variable is treated separately
            for now.
        count : bool, optional
            If False, treats count variables (if present) as continuous.  This
            is the default.  Else if True, the marginal effect is the
            change in probabilities when each observation is increased by one.

        Returns
        -------
        effects : ndarray
            the marginal effect corresponding to the input options

        Notes
        -----
        When using after Poisson, returns the expected number of events
        per period, assuming that the model is loglinear.
        """
        if self.model.constraint is not None:
            warnings.warn('marginal effects ignore constraints', ValueWarning)
        return GEEMargins(self, (at, method, atexog, dummy, count))

    def plot_isotropic_dependence(self, ax=None, xpoints=10, min_n=50):
        """
        Create a plot of the pairwise products of within-group
        residuals against the corresponding time differences.  This
        plot can be used to assess the possible form of an isotropic
        covariance structure.

        Parameters
        ----------
        ax : AxesSubplot
            An axes on which to draw the graph.  If None, new
            figure and axes objects are created
        xpoints : scalar or array_like
            If scalar, the number of points equally spaced points on
            the time difference axis used to define bins for
            calculating local means.  If an array, the specific points
            that define the bins.
        min_n : int
            The minimum sample size in a bin for the mean residual
            product to be included on the plot.
        """
        from statsmodels.graphics import utils as gutils
        resid = self.model.cluster_list(self.resid)
        time = self.model.cluster_list(self.model.time)
        xre, xdt = ([], [])
        for re, ti in zip(resid, time):
            ix = np.tril_indices(re.shape[0], 0)
            re = re[ix[0]] * re[ix[1]] / self.scale ** 2
            xre.append(re)
            dists = np.sqrt(((ti[ix[0], :] - ti[ix[1], :]) ** 2).sum(1))
            xdt.append(dists)
        xre = np.concatenate(xre)
        xdt = np.concatenate(xdt)
        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()
        ii = np.flatnonzero(xdt == 0)
        v0 = np.mean(xre[ii])
        xre /= v0
        if np.isscalar(xpoints):
            xpoints = np.linspace(0, max(xdt), xpoints)
        dg = np.digitize(xdt, xpoints)
        dgu = np.unique(dg)
        hist = np.asarray([np.sum(dg == k) for k in dgu])
        ii = np.flatnonzero(hist >= min_n)
        dgu = dgu[ii]
        dgy = np.asarray([np.mean(xre[dg == k]) for k in dgu])
        dgx = np.asarray([np.mean(xdt[dg == k]) for k in dgu])
        ax.plot(dgx, dgy, '-', color='orange', lw=5)
        ax.set_xlabel('Time difference')
        ax.set_ylabel('Product of scaled residuals')
        return fig

    def sensitivity_params(self, dep_params_first, dep_params_last, num_steps):
        """
        Refits the GEE model using a sequence of values for the
        dependence parameters.

        Parameters
        ----------
        dep_params_first : array_like
            The first dep_params in the sequence
        dep_params_last : array_like
            The last dep_params in the sequence
        num_steps : int
            The number of dep_params in the sequence

        Returns
        -------
        results : array_like
            The GEEResults objects resulting from the fits.
        """
        model = self.model
        import copy
        cov_struct = copy.deepcopy(self.model.cov_struct)
        update_dep = model.update_dep
        model.update_dep = False
        dep_params = []
        results = []
        for x in np.linspace(0, 1, num_steps):
            dp = x * dep_params_last + (1 - x) * dep_params_first
            dep_params.append(dp)
            model.cov_struct = copy.deepcopy(cov_struct)
            model.cov_struct.dep_params = dp
            rslt = model.fit(start_params=self.params, ctol=self.ctol, params_niter=self.params_niter, first_dep_update=self.first_dep_update, cov_type=self.cov_type)
            results.append(rslt)
        model.update_dep = update_dep
        return results
    params_sensitivity = sensitivity_params