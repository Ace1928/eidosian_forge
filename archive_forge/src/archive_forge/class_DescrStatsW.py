import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
class DescrStatsW:
    """
    Descriptive statistics and tests with weights for case weights

    Assumes that the data is 1d or 2d with (nobs, nvars) observations in rows,
    variables in columns, and that the same weight applies to each column.

    If degrees of freedom correction is used, then weights should add up to the
    number of observations. ttest also assumes that the sum of weights
    corresponds to the sample size.

    This is essentially the same as replicating each observations by its
    weight, if the weights are integers, often called case or frequency weights.

    Parameters
    ----------
    data : array_like, 1-D or 2-D
        dataset
    weights : None or 1-D ndarray
        weights for each observation, with same length as zero axis of data
    ddof : int
        default ddof=0, degrees of freedom correction used for second moments,
        var, std, cov, corrcoef.
        However, statistical tests are independent of `ddof`, based on the
        standard formulas.

    Examples
    --------

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> x1_2d = 1.0 + np.random.randn(20, 3)
    >>> w1 = np.random.randint(1, 4, 20)
    >>> d1 = DescrStatsW(x1_2d, weights=w1)
    >>> d1.mean
    array([ 1.42739844,  1.23174284,  1.083753  ])
    >>> d1.var
    array([ 0.94855633,  0.52074626,  1.12309325])
    >>> d1.std_mean
    array([ 0.14682676,  0.10878944,  0.15976497])

    >>> tstat, pval, df = d1.ttest_mean(0)
    >>> tstat; pval; df
    array([  9.72165021,  11.32226471,   6.78342055])
    array([  1.58414212e-12,   1.26536887e-14,   2.37623126e-08])
    44.0

    >>> tstat, pval, df = d1.ttest_mean([0, 1, 1])
    >>> tstat; pval; df
    array([ 9.72165021,  2.13019609,  0.52422632])
    array([  1.58414212e-12,   3.87842808e-02,   6.02752170e-01])
    44.0

    # if weights are integers, then asrepeats can be used

    >>> x1r = d1.asrepeats()
    >>> x1r.shape
    ...
    >>> stats.ttest_1samp(x1r, [0, 1, 1])
    ...

    """

    def __init__(self, data, weights=None, ddof=0):
        self.data = np.asarray(data)
        if weights is None:
            self.weights = np.ones(self.data.shape[0])
        else:
            self.weights = np.asarray(weights).astype(float)
            if len(self.weights.shape) > 1 and len(self.weights) > 1:
                self.weights = self.weights.squeeze()
        self.ddof = ddof

    @cache_readonly
    def sum_weights(self):
        """Sum of weights"""
        return self.weights.sum(0)

    @cache_readonly
    def nobs(self):
        """alias for number of observations/cases, equal to sum of weights
        """
        return self.sum_weights

    @cache_readonly
    def sum(self):
        """weighted sum of data"""
        return np.dot(self.data.T, self.weights)

    @cache_readonly
    def mean(self):
        """weighted mean of data"""
        return self.sum / self.sum_weights

    @cache_readonly
    def demeaned(self):
        """data with weighted mean subtracted"""
        return self.data - self.mean

    @cache_readonly
    def sumsquares(self):
        """weighted sum of squares of demeaned data"""
        return np.dot((self.demeaned ** 2).T, self.weights)

    def var_ddof(self, ddof=0):
        """variance of data given ddof

        Parameters
        ----------
        ddof : int, float
            degrees of freedom correction, independent of attribute ddof

        Returns
        -------
        var : float, ndarray
            variance with denominator ``sum_weights - ddof``
        """
        return self.sumsquares / (self.sum_weights - ddof)

    def std_ddof(self, ddof=0):
        """standard deviation of data with given ddof

        Parameters
        ----------
        ddof : int, float
            degrees of freedom correction, independent of attribute ddof

        Returns
        -------
        std : float, ndarray
            standard deviation with denominator ``sum_weights - ddof``
        """
        return np.sqrt(self.var_ddof(ddof=ddof))

    @cache_readonly
    def var(self):
        """variance with default degrees of freedom correction
        """
        return self.sumsquares / (self.sum_weights - self.ddof)

    @cache_readonly
    def _var(self):
        """variance without degrees of freedom correction

        used for statistical tests with controlled ddof
        """
        return self.sumsquares / self.sum_weights

    @cache_readonly
    def std(self):
        """standard deviation with default degrees of freedom correction
        """
        return np.sqrt(self.var)

    @cache_readonly
    def cov(self):
        """weighted covariance of data if data is 2 dimensional

        assumes variables in columns and observations in rows
        uses default ddof
        """
        cov_ = np.dot(self.weights * self.demeaned.T, self.demeaned)
        cov_ /= self.sum_weights - self.ddof
        return cov_

    @cache_readonly
    def corrcoef(self):
        """weighted correlation with default ddof

        assumes variables in columns and observations in rows
        """
        return self.cov / self.std / self.std[:, None]

    @cache_readonly
    def std_mean(self):
        """standard deviation of weighted mean
        """
        std = self.std
        if self.ddof != 0:
            std = std * np.sqrt((self.sum_weights - self.ddof) / self.sum_weights)
        return std / np.sqrt(self.sum_weights - 1)

    def quantile(self, probs, return_pandas=True):
        """
        Compute quantiles for a weighted sample.

        Parameters
        ----------
        probs : array_like
            A vector of probability points at which to calculate the
            quantiles.  Each element of `probs` should fall in [0, 1].
        return_pandas : bool
            If True, return value is a Pandas DataFrame or Series.
            Otherwise returns a ndarray.

        Returns
        -------
        quantiles : Series, DataFrame, or ndarray
            If `return_pandas` = True, returns one of the following:
              * data are 1d, `return_pandas` = True: a Series indexed by
                the probability points.
              * data are 2d, `return_pandas` = True: a DataFrame with
                the probability points as row index and the variables
                as column index.

            If `return_pandas` = False, returns an ndarray containing the
            same values as the Series/DataFrame.

        Notes
        -----
        To compute the quantiles, first, the weights are summed over
        exact ties yielding distinct data values y_1 < y_2 < ..., and
        corresponding weights w_1, w_2, ....  Let s_j denote the sum
        of the first j weights, and let W denote the sum of all the
        weights.  For a probability point p, if pW falls strictly
        between s_j and s_{j+1} then the estimated quantile is
        y_{j+1}.  If pW = s_j then the estimated quantile is (y_j +
        y_{j+1})/2.  If pW < p_1 then the estimated quantile is y_1.

        References
        ----------
        SAS documentation for weighted quantiles:

        https://support.sas.com/documentation/cdl/en/procstat/63104/HTML/default/viewer.htm#procstat_univariate_sect028.htm
        """
        import pandas as pd
        probs = np.asarray(probs)
        probs = np.atleast_1d(probs)
        if self.data.ndim == 1:
            rslt = self._quantile(self.data, probs)
            if return_pandas:
                rslt = pd.Series(rslt, index=probs)
        else:
            rslt = []
            for vec in self.data.T:
                rslt.append(self._quantile(vec, probs))
            rslt = np.column_stack(rslt)
            if return_pandas:
                columns = ['col%d' % (j + 1) for j in range(rslt.shape[1])]
                rslt = pd.DataFrame(data=rslt, columns=columns, index=probs)
        if return_pandas:
            rslt.index.name = 'p'
        return rslt

    def _quantile(self, vec, probs):
        import pandas as pd
        df = pd.DataFrame(index=np.arange(len(self.weights)))
        df['weights'] = self.weights
        df['vec'] = vec
        dfg = df.groupby('vec').agg('sum')
        weights = dfg.values[:, 0]
        values = np.asarray(dfg.index)
        cweights = np.cumsum(weights)
        totwt = cweights[-1]
        targets = probs * totwt
        ii = np.searchsorted(cweights, targets)
        rslt = values[ii]
        jj = np.flatnonzero(np.abs(targets - cweights[ii]) < 1e-10)
        jj = jj[ii[jj] < len(cweights) - 1]
        rslt[jj] = (values[ii[jj]] + values[ii[jj] + 1]) / 2
        return rslt

    def tconfint_mean(self, alpha=0.05, alternative='two-sided'):
        """two-sided confidence interval for weighted mean of data

        If the data is 2d, then these are separate confidence intervals
        for each column.

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following

              'two-sided': H1: mean not equal to value (default)
              'larger' :   H1: mean larger than value
              'smaller' :  H1: mean smaller than value

        Returns
        -------
        lower, upper : floats or ndarrays
            lower and upper bound of confidence interval

        Notes
        -----
        In a previous version, statsmodels 0.4, alpha was the confidence
        level, e.g. 0.95
        """
        dof = self.sum_weights - 1
        ci = _tconfint_generic(self.mean, self.std_mean, dof, alpha, alternative)
        return ci

    def zconfint_mean(self, alpha=0.05, alternative='two-sided'):
        """two-sided confidence interval for weighted mean of data

        Confidence interval is based on normal distribution.
        If the data is 2d, then these are separate confidence intervals
        for each column.

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following

              'two-sided': H1: mean not equal to value (default)
              'larger' :   H1: mean larger than value
              'smaller' :  H1: mean smaller than value

        Returns
        -------
        lower, upper : floats or ndarrays
            lower and upper bound of confidence interval

        Notes
        -----
        In a previous version, statsmodels 0.4, alpha was the confidence
        level, e.g. 0.95
        """
        return _zconfint_generic(self.mean, self.std_mean, alpha, alternative)

    def ttest_mean(self, value=0, alternative='two-sided'):
        """ttest of Null hypothesis that mean is equal to value.

        The alternative hypothesis H1 is defined by the following

        - 'two-sided': H1: mean not equal to value
        - 'larger' :   H1: mean larger than value
        - 'smaller' :  H1: mean smaller than value

        Parameters
        ----------
        value : float or array
            the hypothesized value for the mean
        alternative : str
            The alternative hypothesis, H1, has to be one of the following:

              - 'two-sided': H1: mean not equal to value (default)
              - 'larger' :   H1: mean larger than value
              - 'smaller' :  H1: mean smaller than value

        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the t-test
        df : int or float

        """
        tstat = (self.mean - value) / self.std_mean
        dof = self.sum_weights - 1
        if alternative == 'two-sided':
            pvalue = stats.t.sf(np.abs(tstat), dof) * 2
        elif alternative == 'larger':
            pvalue = stats.t.sf(tstat, dof)
        elif alternative == 'smaller':
            pvalue = stats.t.cdf(tstat, dof)
        else:
            raise ValueError('alternative not recognized')
        return (tstat, pvalue, dof)

    def ttost_mean(self, low, upp):
        """test of (non-)equivalence of one sample

        TOST: two one-sided t tests

        null hypothesis:  m < low or m > upp
        alternative hypothesis:  low < m < upp

        where m is the expected value of the sample (mean of the population).

        If the pvalue is smaller than a threshold, say 0.05, then we reject the
        hypothesis that the expected value of the sample (mean of the
        population) is outside of the interval given by thresholds low and upp.

        Parameters
        ----------
        low, upp : float
            equivalence interval low < mean < upp

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1, df1 : tuple
            test statistic, pvalue and degrees of freedom for lower threshold
            test
        t2, pv2, df2 : tuple
            test statistic, pvalue and degrees of freedom for upper threshold
            test

        """
        t1, pv1, df1 = self.ttest_mean(low, alternative='larger')
        t2, pv2, df2 = self.ttest_mean(upp, alternative='smaller')
        return (np.maximum(pv1, pv2), (t1, pv1, df1), (t2, pv2, df2))

    def ztest_mean(self, value=0, alternative='two-sided'):
        """z-test of Null hypothesis that mean is equal to value.

        The alternative hypothesis H1 is defined by the following
        'two-sided': H1: mean not equal to value
        'larger' :   H1: mean larger than value
        'smaller' :  H1: mean smaller than value

        Parameters
        ----------
        value : float or array
            the hypothesized value for the mean
        alternative : str
            The alternative hypothesis, H1, has to be one of the following

              'two-sided': H1: mean not equal to value (default)
              'larger' :   H1: mean larger than value
              'smaller' :  H1: mean smaller than value

        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the t-test

        Notes
        -----
        This uses the same degrees of freedom correction as the t-test in the
        calculation of the standard error of the mean, i.e it uses
        `(sum_weights - 1)` instead of `sum_weights` in the denominator.
        See Examples below for the difference.

        Examples
        --------

        z-test on a proportion, with 20 observations, 15 of those are our event

        >>> import statsmodels.api as sm
        >>> x1 = [0, 1]
        >>> w1 = [5, 15]
        >>> d1 = sm.stats.DescrStatsW(x1, w1)
        >>> d1.ztest_mean(0.5)
        (2.5166114784235836, 0.011848940928347452)

        This differs from the proportions_ztest because of the degrees of
        freedom correction:
        >>> sm.stats.proportions_ztest(15, 20.0, value=0.5)
        (2.5819888974716112, 0.009823274507519247).

        We can replicate the results from ``proportions_ztest`` if we increase
        the weights to have artificially one more observation:

        >>> sm.stats.DescrStatsW(x1, np.array(w1)*21./20).ztest_mean(0.5)
        (2.5819888974716116, 0.0098232745075192366)
        """
        tstat = (self.mean - value) / self.std_mean
        if alternative == 'two-sided':
            pvalue = stats.norm.sf(np.abs(tstat)) * 2
        elif alternative == 'larger':
            pvalue = stats.norm.sf(tstat)
        elif alternative == 'smaller':
            pvalue = stats.norm.cdf(tstat)
        return (tstat, pvalue)

    def ztost_mean(self, low, upp):
        """test of (non-)equivalence of one sample, based on z-test

        TOST: two one-sided z-tests

        null hypothesis:  m < low or m > upp
        alternative hypothesis:  low < m < upp

        where m is the expected value of the sample (mean of the population).

        If the pvalue is smaller than a threshold, say 0.05, then we reject the
        hypothesis that the expected value of the sample (mean of the
        population) is outside of the interval given by thresholds low and upp.

        Parameters
        ----------
        low, upp : float
            equivalence interval low < mean < upp

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1 : tuple
            test statistic and p-value for lower threshold test
        t2, pv2 : tuple
            test statistic and p-value for upper threshold test

        """
        t1, pv1 = self.ztest_mean(low, alternative='larger')
        t2, pv2 = self.ztest_mean(upp, alternative='smaller')
        return (np.maximum(pv1, pv2), (t1, pv1), (t2, pv2))

    def get_compare(self, other, weights=None):
        """return an instance of CompareMeans with self and other

        Parameters
        ----------
        other : array_like or instance of DescrStatsW
            If array_like then this creates an instance of DescrStatsW with
            the given weights.
        weights : None or array
            weights are only used if other is not an instance of DescrStatsW

        Returns
        -------
        cm : instance of CompareMeans
            the instance has self attached as d1 and other as d2.

        See Also
        --------
        CompareMeans

        """
        if not isinstance(other, self.__class__):
            d2 = DescrStatsW(other, weights)
        else:
            d2 = other
        return CompareMeans(self, d2)

    def asrepeats(self):
        """get array that has repeats given by floor(weights)

        observations with weight=0 are dropped

        """
        w_int = np.floor(self.weights).astype(int)
        return np.repeat(self.data, w_int, axis=0)