import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
class FTestAnovaPower(Power):
    """Statistical Power calculations F-test for one factor balanced ANOVA

    This is based on Cohen's f as effect size measure.

    See Also
    --------
    statsmodels.stats.oneway.effectsize_oneway

    """

    def power(self, effect_size, nobs, alpha, k_groups=2):
        """Calculate the power of a F-test for one factor ANOVA.

        Parameters
        ----------
        effect_size : float
            standardized effect size. The effect size is here Cohen's f, square
            root of "f2".
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        k_groups : int or float
            number of groups in the ANOVA or k-sample comparison. Default is 2.

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

       """
        return ftest_anova_power(effect_size, nobs, alpha, k_groups=k_groups)

    def solve_power(self, effect_size=None, nobs=None, alpha=None, power=None, k_groups=2):
        """solve for any one parameter of the power of a F-test

        for the one sample F-test the keywords are:
            effect_size, nobs, alpha, power

        Exactly one needs to be ``None``, all others need numeric values.


        Parameters
        ----------
        effect_size : float
            standardized effect size, mean divided by the standard deviation.
            effect size has to be positive.
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

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
        if k_groups is not None:
            self.start_ttp['nobs'] = k_groups * 10
            self.start_bqexp['nobs'] = dict(low=k_groups * 2, start_upp=k_groups * 10)
        if effect_size is None:
            return self._solve_effect_size(effect_size=effect_size, nobs=nobs, alpha=alpha, k_groups=k_groups, power=power)
        return super().solve_power(effect_size=effect_size, nobs=nobs, alpha=alpha, k_groups=k_groups, power=power)

    def _solve_effect_size(self, effect_size=None, nobs=None, alpha=None, power=None, k_groups=2):
        """experimental, test failure in solve_power for effect_size
        """

        def func(x):
            effect_size = x
            return self._power_identity(effect_size=effect_size, nobs=nobs, alpha=alpha, k_groups=k_groups, power=power)
        val, r = optimize.brentq(func, 1e-08, 1 - 1e-08, full_output=True)
        if not r.converged:
            print(r)
        return val