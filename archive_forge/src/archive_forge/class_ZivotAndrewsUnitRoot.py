from __future__ import annotations
from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.compat.python import lzip
from statsmodels.compat.scipy import _next_regular
from typing import Literal, Union
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import correlate
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.validation import (
from statsmodels.tsa._bds import bds
from statsmodels.tsa._innovations import innovations_algo, innovations_filter
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds
class ZivotAndrewsUnitRoot:
    """
    Class wrapper for Zivot-Andrews structural-break unit-root test
    """

    def __init__(self):
        """
        Critical values for the three different models specified for the
        Zivot-Andrews unit-root test.

        Notes
        -----
        The p-values are generated through Monte Carlo simulation using
        100,000 replications and 2000 data points.
        """
        self._za_critical_values = {}
        self._c = ((0.001, -6.78442), (0.1, -5.83192), (0.2, -5.68139), (0.3, -5.58461), (0.4, -5.51308), (0.5, -5.45043), (0.6, -5.39924), (0.7, -5.36023), (0.8, -5.33219), (0.9, -5.30294), (1.0, -5.27644), (2.5, -5.0334), (5.0, -4.81067), (7.5, -4.67636), (10.0, -4.56618), (12.5, -4.4813), (15.0, -4.40507), (17.5, -4.33947), (20.0, -4.28155), (22.5, -4.22683), (25.0, -4.1783), (27.5, -4.13101), (30.0, -4.08586), (32.5, -4.04455), (35.0, -4.0038), (37.5, -3.96144), (40.0, -3.92078), (42.5, -3.88178), (45.0, -3.84503), (47.5, -3.80549), (50.0, -3.77031), (52.5, -3.73209), (55.0, -3.696), (57.5, -3.65985), (60.0, -3.62126), (65.0, -3.5458), (70.0, -3.46848), (75.0, -3.38533), (80.0, -3.29112), (85.0, -3.17832), (90.0, -3.04165), (92.5, -2.95146), (95.0, -2.83179), (96.0, -2.76465), (97.0, -2.68624), (98.0, -2.57884), (99.0, -2.40044), (99.9, -1.88932))
        self._za_critical_values['c'] = np.asarray(self._c)
        self._t = ((0.001, -83.9094), (0.1, -13.8837), (0.2, -9.13205), (0.3, -6.32564), (0.4, -5.60803), (0.5, -5.38794), (0.6, -5.26585), (0.7, -5.18734), (0.8, -5.12756), (0.9, -5.07984), (1.0, -5.03421), (2.5, -4.65634), (5.0, -4.4058), (7.5, -4.25214), (10.0, -4.13678), (12.5, -4.03765), (15.0, -3.95185), (17.5, -3.87945), (20.0, -3.81295), (22.5, -3.75273), (25.0, -3.69836), (27.5, -3.64785), (30.0, -3.59819), (32.5, -3.55146), (35.0, -3.50522), (37.5, -3.45987), (40.0, -3.41672), (42.5, -3.37465), (45.0, -3.33394), (47.5, -3.29393), (50.0, -3.25316), (52.5, -3.21244), (55.0, -3.17124), (57.5, -3.13211), (60.0, -3.09204), (65.0, -3.01135), (70.0, -2.92897), (75.0, -2.83614), (80.0, -2.73893), (85.0, -2.6284), (90.0, -2.49611), (92.5, -2.41337), (95.0, -2.3082), (96.0, -2.25797), (97.0, -2.19648), (98.0, -2.1132), (99.0, -1.99138), (99.9, -1.67466))
        self._za_critical_values['t'] = np.asarray(self._t)
        self._ct = ((0.001, -38.178), (0.1, -6.43107), (0.2, -6.07279), (0.3, -5.95496), (0.4, -5.86254), (0.5, -5.77081), (0.6, -5.72541), (0.7, -5.68406), (0.8, -5.65163), (0.9, -5.60419), (1.0, -5.57556), (2.5, -5.29704), (5.0, -5.07332), (7.5, -4.93003), (10.0, -4.82668), (12.5, -4.73711), (15.0, -4.6602), (17.5, -4.5897), (20.0, -4.52855), (22.5, -4.471), (25.0, -4.42011), (27.5, -4.37387), (30.0, -4.32705), (32.5, -4.28126), (35.0, -4.23793), (37.5, -4.19822), (40.0, -4.158), (42.5, -4.11946), (45.0, -4.08064), (47.5, -4.04286), (50.0, -4.00489), (52.5, -3.96837), (55.0, -3.932), (57.5, -3.89496), (60.0, -3.85577), (65.0, -3.77795), (70.0, -3.69794), (75.0, -3.61852), (80.0, -3.52485), (85.0, -3.41665), (90.0, -3.28527), (92.5, -3.19724), (95.0, -3.08769), (96.0, -3.03088), (97.0, -2.96091), (98.0, -2.85581), (99.0, -2.71015), (99.9, -2.28767))
        self._za_critical_values['ct'] = np.asarray(self._ct)

    def _za_crit(self, stat, model='c'):
        """
        Linear interpolation for Zivot-Andrews p-values and critical values

        Parameters
        ----------
        stat : float
            The ZA test statistic
        model : {"c","t","ct"}
            The model used when computing the ZA statistic. "c" is default.

        Returns
        -------
        pvalue : float
            The interpolated p-value
        cvdict : dict
            Critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        The p-values are linear interpolated from the quantiles of the
        simulated ZA test statistic distribution
        """
        table = self._za_critical_values[model]
        pcnts = table[:, 0]
        stats = table[:, 1]
        pvalue = np.interp(stat, stats, pcnts) / 100.0
        cv = [1.0, 5.0, 10.0]
        crit_value = np.interp(cv, pcnts, stats)
        cvdict = {'1%': crit_value[0], '5%': crit_value[1], '10%': crit_value[2]}
        return (pvalue, cvdict)

    def _quick_ols(self, endog, exog):
        """
        Minimal implementation of LS estimator for internal use
        """
        xpxi = np.linalg.inv(exog.T.dot(exog))
        xpy = exog.T.dot(endog)
        nobs, k_exog = exog.shape
        b = xpxi.dot(xpy)
        e = endog - exog.dot(b)
        sigma2 = e.T.dot(e) / (nobs - k_exog)
        return b / np.sqrt(np.diag(sigma2 * xpxi))

    def _format_regression_data(self, series, nobs, const, trend, cols, lags):
        """
        Create the endog/exog data for the auxiliary regressions
        from the original (standardized) series under test.
        """
        endog = np.diff(series, axis=0)
        endog /= np.sqrt(endog.T.dot(endog))
        series /= np.sqrt(series.T.dot(series))
        exog = np.zeros((endog[lags:].shape[0], cols + lags))
        exog[:, 0] = const
        exog[:, cols - 1] = series[lags:nobs - 1]
        exog[:, cols:] = lagmat(endog, lags, trim='none')[lags:exog.shape[0] + lags]
        return (endog, exog)

    def _update_regression_exog(self, exog, regression, period, nobs, const, trend, cols, lags):
        """
        Update the exog array for the next regression.
        """
        cutoff = period - (lags + 1)
        if regression != 't':
            exog[:cutoff, 1] = 0
            exog[cutoff:, 1] = const
            exog[:, 2] = trend[lags + 2:nobs + 1]
            if regression == 'ct':
                exog[:cutoff, 3] = 0
                exog[cutoff:, 3] = trend[1:nobs - period + 1]
        else:
            exog[:, 1] = trend[lags + 2:nobs + 1]
            exog[:cutoff - 1, 2] = 0
            exog[cutoff - 1:, 2] = trend[0:nobs - period + 1]
        return exog

    def run(self, x, trim=0.15, maxlag=None, regression='c', autolag='AIC'):
        """
        Zivot-Andrews structural-break unit-root test.

        The Zivot-Andrews test tests for a unit root in a univariate process
        in the presence of serial correlation and a single structural break.

        Parameters
        ----------
        x : array_like
            The data series to test.
        trim : float
            The percentage of series at begin/end to exclude from break-period
            calculation in range [0, 0.333] (default=0.15).
        maxlag : int
            The maximum lag which is included in test, default is
            12*(nobs/100)^{1/4} (Schwert, 1989).
        regression : {"c","t","ct"}
            Constant and trend order to include in regression.

            * "c" : constant only (default).
            * "t" : trend only.
            * "ct" : constant and trend.
        autolag : {"AIC", "BIC", "t-stat", None}
            The method to select the lag length when using automatic selection.

            * if None, then maxlag lags are used,
            * if "AIC" (default) or "BIC", then the number of lags is chosen
              to minimize the corresponding information criterion,
            * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
              lag until the t-statistic on the last lag length is significant
              using a 5%-sized test.

        Returns
        -------
        zastat : float
            The test statistic.
        pvalue : float
            The pvalue based on MC-derived critical values.
        cvdict : dict
            The critical values for the test statistic at the 1%, 5%, and 10%
            levels.
        baselag : int
            The number of lags used for period regressions.
        bpidx : int
            The index of x corresponding to endogenously calculated break period
            with values in the range [0..nobs-1].

        Notes
        -----
        H0 = unit root with a single structural break

        Algorithm follows Baum (2004/2015) approximation to original
        Zivot-Andrews method. Rather than performing an autolag regression at
        each candidate break period (as per the original paper), a single
        autolag regression is run up-front on the base model (constant + trend
        with no dummies) to determine the best lag length. This lag length is
        then used for all subsequent break-period regressions. This results in
        significant run time reduction but also slightly more pessimistic test
        statistics than the original Zivot-Andrews method, although no attempt
        has been made to characterize the size/power trade-off.

        References
        ----------
        .. [1] Baum, C.F. (2004). ZANDREWS: Stata module to calculate
           Zivot-Andrews unit root test in presence of structural break,"
           Statistical Software Components S437301, Boston College Department
           of Economics, revised 2015.

        .. [2] Schwert, G.W. (1989). Tests for unit roots: A Monte Carlo
           investigation. Journal of Business & Economic Statistics, 7:
           147-159.

        .. [3] Zivot, E., and Andrews, D.W.K. (1992). Further evidence on the
           great crash, the oil-price shock, and the unit-root hypothesis.
           Journal of Business & Economic Studies, 10: 251-270.
        """
        x = array_like(x, 'x')
        trim = float_like(trim, 'trim')
        maxlag = int_like(maxlag, 'maxlag', optional=True)
        regression = string_like(regression, 'regression', options=('c', 't', 'ct'))
        autolag = string_like(autolag, 'autolag', options=('aic', 'bic', 't-stat'), optional=True)
        if trim < 0 or trim > 1.0 / 3.0:
            raise ValueError('trim value must be a float in range [0, 1/3)')
        nobs = x.shape[0]
        if autolag:
            adf_res = adfuller(x, maxlag=maxlag, regression='ct', autolag=autolag)
            baselags = adf_res[2]
        elif maxlag:
            baselags = maxlag
        else:
            baselags = int(12.0 * np.power(nobs / 100.0, 1 / 4.0))
        trimcnt = int(nobs * trim)
        start_period = trimcnt
        end_period = nobs - trimcnt
        if regression == 'ct':
            basecols = 5
        else:
            basecols = 4
        c_const = 1 / np.sqrt(nobs)
        t_const = np.arange(1.0, nobs + 2)
        t_const *= np.sqrt(3) / nobs ** (3 / 2)
        endog, exog = self._format_regression_data(x, nobs, c_const, t_const, basecols, baselags)
        stats = np.full(end_period + 1, np.inf)
        for bp in range(start_period + 1, end_period + 1):
            exog = self._update_regression_exog(exog, regression, bp, nobs, c_const, t_const, basecols, baselags)
            if bp == start_period + 1:
                o = OLS(endog[baselags:], exog, hasconst=1).fit()
                if o.df_model < exog.shape[1] - 1:
                    raise ValueError('ZA: auxiliary exog matrix is not full rank.\n  cols (exc intercept) = {}  rank = {}'.format(exog.shape[1] - 1, o.df_model))
                stats[bp] = o.tvalues[basecols - 1]
            else:
                stats[bp] = self._quick_ols(endog[baselags:], exog)[basecols - 1]
        zastat = np.min(stats)
        bpidx = np.argmin(stats) - 1
        crit = self._za_crit(zastat, regression)
        pval = crit[0]
        cvdict = crit[1]
        return (zastat, pval, cvdict, baselags, bpidx)

    def __call__(self, x, trim=0.15, maxlag=None, regression='c', autolag='AIC'):
        return self.run(x, trim=trim, maxlag=maxlag, regression=regression, autolag=autolag)