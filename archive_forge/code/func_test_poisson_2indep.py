import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def test_poisson_2indep(count1, exposure1, count2, exposure2, value=None, ratio_null=None, method=None, compare='ratio', alternative='two-sided', etest_kwds=None):
    """Test for comparing two sample Poisson intensity rates.

    Rates are defined as expected count divided by exposure.

    The Null and alternative hypothesis for the rates, rate1 and rate2, of two
    independent Poisson samples are

    for compare = 'diff'

    - H0: rate1 - rate2 - value = 0
    - H1: rate1 - rate2 - value != 0  if alternative = 'two-sided'
    - H1: rate1 - rate2 - value > 0   if alternative = 'larger'
    - H1: rate1 - rate2 - value < 0   if alternative = 'smaller'

    for compare = 'ratio'

    - H0: rate1 / rate2 - value = 0
    - H1: rate1 / rate2 - value != 0  if alternative = 'two-sided'
    - H1: rate1 / rate2 - value > 0   if alternative = 'larger'
    - H1: rate1 / rate2 - value < 0   if alternative = 'smaller'

    Parameters
    ----------
    count1 : int
        Number of events in first sample, treatment group.
    exposure1 : float
        Total exposure (time * subjects) in first sample.
    count2 : int
        Number of events in second sample, control group.
    exposure2 : float
        Total exposure (time * subjects) in second sample.
    ratio_null: float
        Ratio of the two Poisson rates under the Null hypothesis. Default is 1.
        Deprecated, use ``value`` instead.

        .. deprecated:: 0.14.0

            Use ``value`` instead.

    value : float
        Value of the ratio or difference of 2 independent rates under the null
        hypothesis. Default is equal rates, i.e. 1 for ratio and 0 for diff.

        .. versionadded:: 0.14.0

            Replacement for ``ratio_null``.

    method : string
        Method for the test statistic and the p-value. Defaults to `'score'`.
        see Notes.

        ratio:

        - 'wald': method W1A, wald test, variance based on observed rates
        - 'score': method W2A, score test, variance based on estimate under
          the Null hypothesis
        - 'wald-log': W3A, uses log-ratio, variance based on observed rates
        - 'score-log' W4A, uses log-ratio, variance based on estimate under
          the Null hypothesis
        - 'sqrt': W5A, based on variance stabilizing square root transformation
        - 'exact-cond': exact conditional test based on binomial distribution
           This uses ``binom_test`` which is minlike in the two-sided case.
        - 'cond-midp': midpoint-pvalue of exact conditional test
        - 'etest' or 'etest-score: etest with score test statistic
        - 'etest-wald': etest with wald test statistic

        diff:

        - 'wald',
        - 'waldccv'
        - 'score'
        - 'etest-score' or 'etest: etest with score test statistic
        - 'etest-wald': etest with wald test statistic

    compare : {'diff', 'ratio'}
        Default is "ratio".
        If compare is `ratio`, then the hypothesis test is for the
        rate ratio defined by ratio = rate1 / rate2.
        If compare is `diff`, then the hypothesis test is for
        diff = rate1 - rate2.
    alternative : {"two-sided" (default), "larger", smaller}
        The alternative hypothesis, H1, has to be one of the following

        - 'two-sided': H1: ratio, or diff, of rates is not equal to value
        - 'larger' :   H1: ratio, or diff, of rates is larger than value
        - 'smaller' :  H1: ratio, or diff, of rates is smaller than value
    etest_kwds: dictionary
        Additional optional parameters to be passed to the etest_poisson_2indep
        function, namely y_grid.

    Returns
    -------
    results : instance of HolderTuple class
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`.

    See Also
    --------
    tost_poisson_2indep
    etest_poisson_2indep

    Notes
    -----
    The hypothesis tests for compare="ratio" are based on Gu et al 2018.
    The e-tests are also based on ...

    - 'wald': method W1A, wald test, variance based on separate estimates
    - 'score': method W2A, score test, variance based on estimate under Null
    - 'wald-log': W3A, wald test for log transformed ratio
    - 'score-log' W4A, score test for log transformed ratio
    - 'sqrt': W5A, based on variance stabilizing square root transformation
    - 'exact-cond': exact conditional test based on binomial distribution
    - 'cond-midp': midpoint-pvalue of exact conditional test
    - 'etest': etest with score test statistic
    - 'etest-wald': etest with wald test statistic

    The hypothesis test for compare="diff" are mainly based on Ng et al 2007
    and ...

    - wald
    - score
    - etest-score
    - etest-wald

    Note the etests use the constraint maximum likelihood estimate (cmle) as
    parameters for the underlying Poisson probabilities. The constraint cmle
    parameters are the same as in the score test.
    The E-test in Krishnamoorty and Thomson uses a moment estimator instead of
    the score estimator.

    References
    ----------
    .. [1] Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
       Biometrical Journal 50 (2008) 2, 2008

    .. [2] Ng, H. K. T., K. Gu, and M. L. Tang. 2007. “A Comparative Study of
       Tests for the Difference of Two Poisson Means.”
       Computational Statistics & Data Analysis 51 (6): 3085–99.
       https://doi.org/10.1016/j.csda.2006.02.004.

    """
    y1, n1, y2, n2 = map(np.asarray, [count1, exposure1, count2, exposure2])
    d = n2 / n1
    rate1, rate2 = (y1 / n1, y2 / n2)
    rates_cmle = None
    if compare == 'ratio':
        if method is None:
            method = 'score'
        if ratio_null is not None:
            warnings.warn("'ratio_null' is deprecated, use 'value' keyword", FutureWarning)
            value = ratio_null
        if ratio_null is None and value is None:
            value = ratio_null = 1
        else:
            ratio_null = value
        r = value
        r_d = r / d
        if method in ['score']:
            stat = (y1 - y2 * r_d) / np.sqrt((y1 + y2) * r_d)
            dist = 'normal'
        elif method in ['wald']:
            stat = (y1 - y2 * r_d) / np.sqrt(y1 + y2 * r_d ** 2)
            dist = 'normal'
        elif method in ['score-log']:
            stat = np.log(y1 / y2) - np.log(r_d)
            stat /= np.sqrt((2 + 1 / r_d + r_d) / (y1 + y2))
            dist = 'normal'
        elif method in ['wald-log']:
            stat = (np.log(y1 / y2) - np.log(r_d)) / np.sqrt(1 / y1 + 1 / y2)
            dist = 'normal'
        elif method in ['sqrt']:
            stat = 2 * (np.sqrt(y1 + 3 / 8.0) - np.sqrt((y2 + 3 / 8.0) * r_d))
            stat /= np.sqrt(1 + r_d)
            dist = 'normal'
        elif method in ['exact-cond', 'cond-midp']:
            from statsmodels.stats import proportion
            bp = r_d / (1 + r_d)
            y_total = y1 + y2
            stat = np.nan
            pvalue = proportion.binom_test(y1, y_total, prop=bp, alternative=alternative)
            if method in ['cond-midp']:
                pvalue = pvalue - 0.5 * stats.binom.pmf(y1, y_total, bp)
            dist = 'binomial'
        elif method.startswith('etest'):
            if method.endswith('wald'):
                method_etest = 'wald'
            else:
                method_etest = 'score'
            if etest_kwds is None:
                etest_kwds = {}
            stat, pvalue = etest_poisson_2indep(count1, exposure1, count2, exposure2, value=value, method=method_etest, alternative=alternative, **etest_kwds)
            dist = 'poisson'
        else:
            raise ValueError(f'method "{method}" not recognized')
    elif compare == 'diff':
        if value is None:
            value = 0
        if method in ['wald']:
            stat = (rate1 - rate2 - value) / np.sqrt(rate1 / n1 + rate2 / n2)
            dist = 'normal'
            'waldccv'
        elif method in ['waldccv']:
            stat = rate1 - rate2 - value
            stat /= np.sqrt((count1 + 0.5) / n1 ** 2 + (count2 + 0.5) / n2 ** 2)
            dist = 'normal'
        elif method in ['score']:
            count_pooled = y1 + y2
            rate_pooled = count_pooled / (n1 + n2)
            dt = rate_pooled - value
            r2_cmle = 0.5 * (dt + np.sqrt(dt ** 2 + 4 * value * y2 / (n1 + n2)))
            r1_cmle = r2_cmle + value
            stat = (rate1 - rate2 - value) / np.sqrt(r1_cmle / n1 + r2_cmle / n2)
            rates_cmle = (r1_cmle, r2_cmle)
            dist = 'normal'
        elif method.startswith('etest'):
            if method.endswith('wald'):
                method_etest = 'wald'
            else:
                method_etest = 'score'
                if method == 'etest':
                    method = method + '-score'
            if etest_kwds is None:
                etest_kwds = {}
            stat, pvalue = etest_poisson_2indep(count1, exposure1, count2, exposure2, value=value, method=method_etest, compare='diff', alternative=alternative, **etest_kwds)
            dist = 'poisson'
        else:
            raise ValueError(f'method "{method}" not recognized')
    else:
        raise NotImplementedError('"compare" needs to be ratio or diff')
    if dist == 'normal':
        stat, pvalue = _zstat_generic2(stat, 1, alternative)
    rates = (rate1, rate2)
    ratio = rate1 / rate2
    diff = rate1 - rate2
    res = HolderTuple(statistic=stat, pvalue=pvalue, distribution=dist, compare=compare, method=method, alternative=alternative, rates=rates, ratio=ratio, diff=diff, value=value, rates_cmle=rates_cmle, ratio_null=ratio_null)
    return res