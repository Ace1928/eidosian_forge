import numpy as np
from scipy import special
from statsmodels.stats.base import Holder
noncentrality parameter for t statistic

    Parameters
    ----------
    fstat : float
        f-statistic, for example from a hypothesis test
        df : int or float
        Degrees of freedom
    alpha : float in (0, 1)
        Significance level for the confidence interval, covarage is 1 - alpha.

    Returns
    -------
    HolderTuple
        The main attributes are

        - ``nc`` : estimate of noncentrality parameter
        - ``confint`` : lower and upper bound of confidence interval for `nc``

        Other attributes are estimates for nc by different methods.

    References
    ----------
    .. [1] Hedges, Larry V. 2016. “Distribution Theory for Glass’s Estimator of
       Effect Size and Related Estimators:”
       Journal of Educational Statistics, November.
       https://doi.org/10.3102/10769986006002107.

    