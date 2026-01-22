import warnings
import math
from math import gcd
from collections import namedtuple
import numpy as np
from numpy import array, asarray, ma
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements
from scipy._lib._util import (check_random_state, MapWrapper, _get_nan,
import scipy.special as special
from scipy import linalg
from . import distributions
from . import _mstats_basic as mstats_basic
from ._stats_mstats_common import (_find_repeats, linregress, theilslopes,
from ._stats import (_kendall_dis, _toint64, _weightedrankedtau,
from dataclasses import dataclass, field
from ._hypotests import _all_partitions
from ._stats_pythran import _compute_outer_prob_inside_method
from ._resampling import (MonteCarloMethod, PermutationMethod, BootstrapMethod,
from ._axis_nan_policy import (_axis_nan_policy_factory,
from ._binomtest import _binary_search_for_binom_tst as _binary_search
from scipy._lib._bunch import _make_tuple_bunch
from scipy import stats
from scipy.optimize import root_scalar
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy._lib._util import normalize_axis_index
from scipy._lib._util import float_factorial  # noqa: F401
from scipy.stats._mstats_basic import (  # noqa: F401
@dataclass
class QuantileTestResult:
    """
    Result of `scipy.stats.quantile_test`.

    Attributes
    ----------
    statistic: float
        The statistic used to calculate the p-value; either ``T1``, the
        number of observations less than or equal to the hypothesized quantile,
        or ``T2``, the number of observations strictly less than the
        hypothesized quantile. Two test statistics are required to handle the
        possibility the data was generated from a discrete or mixed
        distribution.

    statistic_type : int
        ``1`` or ``2`` depending on which of ``T1`` or ``T2`` was used to
        calculate the p-value respectively. ``T1`` corresponds to the
        ``"greater"`` alternative hypothesis and ``T2`` to the ``"less"``.  For
        the ``"two-sided"`` case, the statistic type that leads to smallest
        p-value is used.  For significant tests, ``statistic_type = 1`` means
        there is evidence that the population quantile is significantly greater
        than the hypothesized value and ``statistic_type = 2`` means there is
        evidence that it is significantly less than the hypothesized value.

    pvalue : float
        The p-value of the hypothesis test.
    """
    statistic: float
    statistic_type: int
    pvalue: float
    _alternative: list[str] = field(repr=False)
    _x: np.ndarray = field(repr=False)
    _p: float = field(repr=False)

    def confidence_interval(self, confidence_level=0.95):
        """
        Compute the confidence interval of the quantile.

        Parameters
        ----------
        confidence_level : float, default: 0.95
            Confidence level for the computed confidence interval
            of the quantile. Default is 0.95.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence interval.

        Examples
        --------
        >>> import numpy as np
        >>> import scipy.stats as stats
        >>> p = 0.75  # quantile of interest
        >>> q = 0  # hypothesized value of the quantile
        >>> x = np.exp(np.arange(0, 1.01, 0.01))
        >>> res = stats.quantile_test(x, q=q, p=p, alternative='less')
        >>> lb, ub = res.confidence_interval()
        >>> lb, ub
        (-inf, 2.293318740264183)
        >>> res = stats.quantile_test(x, q=q, p=p, alternative='two-sided')
        >>> lb, ub = res.confidence_interval(0.9)
        >>> lb, ub
        (1.9542373206359396, 2.293318740264183)
        """
        alternative = self._alternative
        p = self._p
        x = np.sort(self._x)
        n = len(x)
        bd = stats.binom(n, p)
        if confidence_level <= 0 or confidence_level >= 1:
            message = '`confidence_level` must be a number between 0 and 1.'
            raise ValueError(message)
        low_index = np.nan
        high_index = np.nan
        if alternative == 'less':
            p = 1 - confidence_level
            low = -np.inf
            high_index = int(bd.isf(p))
            high = x[high_index] if high_index < n else np.nan
        elif alternative == 'greater':
            p = 1 - confidence_level
            low_index = int(bd.ppf(p)) - 1
            low = x[low_index] if low_index >= 0 else np.nan
            high = np.inf
        elif alternative == 'two-sided':
            p = (1 - confidence_level) / 2
            low_index = int(bd.ppf(p)) - 1
            low = x[low_index] if low_index >= 0 else np.nan
            high_index = int(bd.isf(p))
            high = x[high_index] if high_index < n else np.nan
        return ConfidenceInterval(low, high)