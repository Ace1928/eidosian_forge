import numbers
import numpy as np
def ttest_mean(self, value=0, transform='trimmed', alternative='two-sided'):
    """
        One sample t-test for trimmed or Winsorized mean

        Parameters
        ----------
        value : float
            Value of the mean under the Null hypothesis
        transform : {'trimmed', 'winsorized'}
            Specified whether the mean test is based on trimmed or winsorized
            data.
        alternative : {'two-sided', 'larger', 'smaller'}


        Notes
        -----
        p-value is based on the approximate t-distribution of the test
        statistic. The approximation is valid if the underlying distribution
        is symmetric.
        """
    import statsmodels.stats.weightstats as smws
    df = self.nobs_reduced - 1
    if transform == 'trimmed':
        mean_ = self.mean_trimmed
        std_ = self.std_mean_trimmed
    elif transform == 'winsorized':
        mean_ = self.mean_winsorized
        std_ = self.std_mean_winsorized
    else:
        raise ValueError("transform can only be 'trimmed' or 'winsorized'")
    res = smws._tstat_generic(mean_, 0, std_, df, alternative=alternative, diff=value)
    return res + (df,)