import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.base import HolderTuple
class CombineResults:
    """Results from combined estimate of means or effect sizes

    This currently includes intermediate results that might be removed
    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        self._ini_keys = list(kwds.keys())
        self.df_resid = self.k - 1
        self.sd_eff_w_fe_hksj = np.sqrt(self.var_hksj_fe)
        self.sd_eff_w_re_hksj = np.sqrt(self.var_hksj_re)
        self.h2 = self.q / (self.k - 1)
        self.i2 = 1 - 1 / self.h2
        self.cache_ci = {}

    def conf_int_samples(self, alpha=0.05, use_t=None, nobs=None, ci_func=None):
        """confidence intervals for the effect size estimate of samples

        Additional information needs to be provided for confidence intervals
        that are not based on normal distribution using available variance.
        This is likely to change in future.

        Parameters
        ----------
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.
        nobs : None or float
            Number of observations used for degrees of freedom computation.
            Only used if use_t is true.
        ci_func : None or callable
            User provided function to compute confidence intervals.
            This is not used yet and will allow using non-standard confidence
            intervals.

        Returns
        -------
        ci_eff : tuple of ndarrays
            Tuple (ci_low, ci_upp) with confidence interval computed for each
            sample.

        Notes
        -----
        CombineResults currently only has information from the combine_effects
        function, which does not provide details about individual samples.
        """
        if (alpha, use_t) in self.cache_ci:
            return self.cache_ci[alpha, use_t]
        if use_t is None:
            use_t = self.use_t
        if ci_func is not None:
            kwds = {'use_t': use_t} if use_t is not None else {}
            ci_eff = ci_func(alpha=alpha, **kwds)
            self.ci_sample_distr = 'ci_func'
        else:
            if use_t is False:
                crit = stats.norm.isf(alpha / 2)
                self.ci_sample_distr = 'normal'
            elif nobs is not None:
                df_resid = nobs - 1
                crit = stats.t.isf(alpha / 2, df_resid)
                self.ci_sample_distr = 't'
            else:
                msg = '`use_t=True` requires `nobs` for each sample or `ci_func`. Using normal distribution for confidence interval of individual samples.'
                import warnings
                warnings.warn(msg)
                crit = stats.norm.isf(alpha / 2)
                self.ci_sample_distr = 'normal'
            ci_low = self.eff - crit * self.sd_eff
            ci_upp = self.eff + crit * self.sd_eff
            ci_eff = (ci_low, ci_upp)
        self.cache_ci[alpha, use_t] = ci_eff
        return ci_eff

    def conf_int(self, alpha=0.05, use_t=None):
        """confidence interval for the overall mean estimate

        Parameters
        ----------
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.

        Returns
        -------
        ci_eff_fe : tuple of floats
            Confidence interval for mean effects size based on fixed effects
            model with scale=1.
        ci_eff_re : tuple of floats
            Confidence interval for mean effects size based on random effects
            model with scale=1
        ci_eff_fe_wls : tuple of floats
            Confidence interval for mean effects size based on fixed effects
            model with estimated scale corresponding to WLS, ie. HKSJ.
        ci_eff_re_wls : tuple of floats
            Confidence interval for mean effects size based on random effects
            model with estimated scale corresponding to WLS, ie. HKSJ.
            If random effects method is fully iterated, i.e. Paule-Mandel, then
            the estimated scale is 1.

        """
        if use_t is None:
            use_t = self.use_t
        if use_t is False:
            crit = stats.norm.isf(alpha / 2)
        else:
            crit = stats.t.isf(alpha / 2, self.df_resid)
        sgn = np.asarray([-1, 1])
        m_fe = self.mean_effect_fe
        m_re = self.mean_effect_re
        ci_eff_fe = m_fe + sgn * crit * self.sd_eff_w_fe
        ci_eff_re = m_re + sgn * crit * self.sd_eff_w_re
        ci_eff_fe_wls = m_fe + sgn * crit * np.sqrt(self.var_hksj_fe)
        ci_eff_re_wls = m_re + sgn * crit * np.sqrt(self.var_hksj_re)
        return (ci_eff_fe, ci_eff_re, ci_eff_fe_wls, ci_eff_re_wls)

    def test_homogeneity(self):
        """Test whether the means of all samples are the same

        currently no options, test uses chisquare distribution
        default might change depending on `use_t`

        Returns
        -------
        res : HolderTuple instance
            The results include the following attributes:

            - statistic : float
                Test statistic, ``q`` in meta-analysis, this is the
                pearson_chi2 statistic for the fixed effects model.
            - pvalue : float
                P-value based on chisquare distribution.
            - df : float
                Degrees of freedom, equal to number of studies or samples
                minus 1.
        """
        pvalue = stats.chi2.sf(self.q, self.k - 1)
        res = HolderTuple(statistic=self.q, pvalue=pvalue, df=self.k - 1, distr='chi2')
        return res

    def summary_array(self, alpha=0.05, use_t=None):
        """Create array with sample statistics and mean estimates

        Parameters
        ----------
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.

        Returns
        -------
        res : ndarray
            Array with columns
            ['eff', "sd_eff", "ci_low", "ci_upp", "w_fe","w_re"].
            Rows include statistics for samples and estimates of overall mean.
        column_names : list of str
            The names for the columns, used when creating summary DataFrame.
        """
        ci_low, ci_upp = self.conf_int_samples(alpha=alpha, use_t=use_t)
        res = np.column_stack([self.eff, self.sd_eff, ci_low, ci_upp, self.weights_rel_fe, self.weights_rel_re])
        ci = self.conf_int(alpha=alpha, use_t=use_t)
        res_fe = [[self.mean_effect_fe, self.sd_eff_w_fe, ci[0][0], ci[0][1], 1, np.nan]]
        res_re = [[self.mean_effect_re, self.sd_eff_w_re, ci[1][0], ci[1][1], np.nan, 1]]
        res_fe_wls = [[self.mean_effect_fe, self.sd_eff_w_fe_hksj, ci[2][0], ci[2][1], 1, np.nan]]
        res_re_wls = [[self.mean_effect_re, self.sd_eff_w_re_hksj, ci[3][0], ci[3][1], np.nan, 1]]
        res = np.concatenate([res, res_fe, res_re, res_fe_wls, res_re_wls], axis=0)
        column_names = ['eff', 'sd_eff', 'ci_low', 'ci_upp', 'w_fe', 'w_re']
        return (res, column_names)

    def summary_frame(self, alpha=0.05, use_t=None):
        """Create DataFrame with sample statistics and mean estimates

        Parameters
        ----------
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.

        Returns
        -------
        res : DataFrame
            pandas DataFrame instance with columns
            ['eff', "sd_eff", "ci_low", "ci_upp", "w_fe","w_re"].
            Rows include statistics for samples and estimates of overall mean.

        """
        if use_t is None:
            use_t = self.use_t
        labels = list(self.row_names) + ['fixed effect', 'random effect', 'fixed effect wls', 'random effect wls']
        res, col_names = self.summary_array(alpha=alpha, use_t=use_t)
        results = pd.DataFrame(res, index=labels, columns=col_names)
        return results

    def plot_forest(self, alpha=0.05, use_t=None, use_exp=False, ax=None, **kwds):
        """Forest plot with means and confidence intervals

        Parameters
        ----------
        ax : None or matplotlib axis instance
            If ax is provided, then the plot will be added to it.
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.
        use_exp : bool
            If `use_exp` is True, then the effect size and confidence limits
            will be exponentiated. This transform log-odds-ration into
            odds-ratio, and similarly for risk-ratio.
        ax : AxesSubplot, optional
            If given, this axes is used to plot in instead of a new figure
            being created.
        kwds : optional keyword arguments
            Keywords are forwarded to the dot_plot function that creates the
            plot.

        Returns
        -------
        fig : Matplotlib figure instance

        See Also
        --------
        dot_plot

        """
        from statsmodels.graphics.dotplots import dot_plot
        res_df = self.summary_frame(alpha=alpha, use_t=use_t)
        if use_exp:
            res_df = np.exp(res_df[['eff', 'ci_low', 'ci_upp']])
        hw = np.abs(res_df[['ci_low', 'ci_upp']] - res_df[['eff']].values)
        fig = dot_plot(points=res_df['eff'], intervals=hw, lines=res_df.index, line_order=res_df.index, **kwds)
        return fig