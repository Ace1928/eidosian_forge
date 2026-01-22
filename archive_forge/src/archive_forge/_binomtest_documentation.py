from math import sqrt
import numpy as np
from scipy._lib._util import _validate_int
from scipy.optimize import brentq
from scipy.special import ndtri
from ._discrete_distns import binom
from ._common import ConfidenceInterval

        Compute the confidence interval for ``statistic``.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval
            of the estimated proportion. Default is 0.95.
        method : {'exact', 'wilson', 'wilsoncc'}, optional
            Selects the method used to compute the confidence interval
            for the estimate of the proportion:

            'exact' :
                Use the Clopper-Pearson exact method [1]_.
            'wilson' :
                Wilson's method, without continuity correction ([2]_, [3]_).
            'wilsoncc' :
                Wilson's method, with continuity correction ([2]_, [3]_).

            Default is ``'exact'``.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence interval.

        References
        ----------
        .. [1] C. J. Clopper and E. S. Pearson, The use of confidence or
               fiducial limits illustrated in the case of the binomial,
               Biometrika, Vol. 26, No. 4, pp 404-413 (Dec. 1934).
        .. [2] E. B. Wilson, Probable inference, the law of succession, and
               statistical inference, J. Amer. Stat. Assoc., 22, pp 209-212
               (1927).
        .. [3] Robert G. Newcombe, Two-sided confidence intervals for the
               single proportion: comparison of seven methods, Statistics
               in Medicine, 17, pp 857-872 (1998).

        Examples
        --------
        >>> from scipy.stats import binomtest
        >>> result = binomtest(k=7, n=50, p=0.1)
        >>> result.statistic
        0.14
        >>> result.proportion_ci()
        ConfidenceInterval(low=0.05819170033997342, high=0.26739600249700846)
        