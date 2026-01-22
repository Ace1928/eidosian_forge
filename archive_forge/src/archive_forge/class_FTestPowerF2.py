import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
class FTestPowerF2(Power):
    """Statistical Power calculations for generic F-test of a constraint

    This is based on Cohen's f^2 as effect size measure.

    Examples
    --------
    Sample size and power for multiple regression base on R-squared

    Compute effect size from R-squared

    >>> r2 = 0.1
    >>> f2 = r2 / (1 - r2)
    >>> f = np.sqrt(f2)
    >>> r2, f2, f
    (0.1, 0.11111111111111112, 0.33333333333333337)

    Find sample size by solving for denominator degrees of freedom.

    >>> df1 = 1  # number of constraints in hypothesis test
    >>> df2 = FTestPowerF2().solve_power(effect_size=f2, alpha=0.1, power=0.9,
                                         df_num=df1)
    >>> ncc = 1  # default
    >>> nobs = df2 + df1 + ncc
    >>> df2, nobs
    (76.46459758305376, 78.46459758305376)

    verify power at df2

    >>> FTestPowerF2().power(effect_size=f, alpha=0.1, df_num=df1, df_denom=df2)
    0.8999999972109698

    """

    def power(self, effect_size, df_num, df_denom, alpha, ncc=1):
        """Calculate the power of a F-test.

        The effect size is Cohen's ``f^2``.

        The sample size is given by ``nobs = df_denom + df_num + ncc``

        Parameters
        ----------
        effect_size : float
            The effect size is here Cohen's ``f2``. This is equal to
            the noncentrality of an F-test divided by nobs.
        df_num : int or float
            Numerator degrees of freedom,
            This corresponds to the number of constraints in Wald tests.
        df_denom : int or float
            Denominator degrees of freedom.
            This corresponds to the df_resid in Wald tests.
        alpha : float in interval (0,1)
            Significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        ncc : int
            Degrees of freedom correction for non-centrality parameter.
            see Notes

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Notes
        -----
        The sample size is given implicitly by df_denom

        set ncc=0 to match t-test, or f-test in LikelihoodModelResults.
        ncc=1 matches the non-centrality parameter in R::pwr::pwr.f2.test

        ftest_power with ncc=0 should also be correct for f_test in regression
        models, with df_num and d_denom as defined there. (not verified yet)
        """
        pow_ = ftest_power_f2(effect_size, df_num, df_denom, alpha, ncc=ncc)
        return pow_

    def solve_power(self, effect_size=None, df_num=None, df_denom=None, alpha=None, power=None, ncc=1):
        """Solve for any one parameter of the power of a F-test

        for the one sample F-test the keywords are:
            effect_size, df_num, df_denom, alpha, power

        Exactly one needs to be ``None``, all others need numeric values.

        The effect size is Cohen's ``f2``.

        The sample size is given by ``nobs = df_denom + df_num + ncc``, and
        can be found by solving for df_denom.

        Parameters
        ----------
        effect_size : float
            The effect size is here Cohen's ``f2``. This is equal to
            the noncentrality of an F-test divided by nobs.
        df_num : int or float
            Numerator degrees of freedom,
            This corresponds to the number of constraints in Wald tests.
        df_denom : int or float
            Denominator degrees of freedom.
            This corresponds to the df_resid in Wald tests.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        ncc : int
            degrees of freedom correction for non-centrality parameter.
            see Notes

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        """
        return super().solve_power(effect_size=effect_size, df_num=df_num, df_denom=df_denom, alpha=alpha, power=power, ncc=ncc)