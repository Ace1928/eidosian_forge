from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import warnings
import numpy as np
from scipy import special, interpolate, stats
from scipy.stats._censored_data import CensoredData
from scipy.stats._common import ConfidenceInterval
@dataclass
class EmpiricalDistributionFunction:
    """An empirical distribution function produced by `scipy.stats.ecdf`

    Attributes
    ----------
    quantiles : ndarray
        The unique values of the sample from which the
        `EmpiricalDistributionFunction` was estimated.
    probabilities : ndarray
        The point estimates of the cumulative distribution function (CDF) or
        its complement, the survival function (SF), corresponding with
        `quantiles`.
    """
    quantiles: np.ndarray
    probabilities: np.ndarray
    _n: np.ndarray = field(repr=False)
    _d: np.ndarray = field(repr=False)
    _sf: np.ndarray = field(repr=False)
    _kind: str = field(repr=False)

    def __init__(self, q, p, n, d, kind):
        self.probabilities = p
        self.quantiles = q
        self._n = n
        self._d = d
        self._sf = p if kind == 'sf' else 1 - p
        self._kind = kind
        f0 = 1 if kind == 'sf' else 0
        f1 = 1 - f0
        x = np.insert(q, [0, len(q)], [-np.inf, np.inf])
        y = np.insert(p, [0, len(p)], [f0, f1])
        self._f = interpolate.interp1d(x, y, kind='previous', assume_sorted=True)

    def evaluate(self, x):
        """Evaluate the empirical CDF/SF function at the input.

        Parameters
        ----------
        x : ndarray
            Argument to the CDF/SF

        Returns
        -------
        y : ndarray
            The CDF/SF evaluated at the input
        """
        return self._f(x)

    def plot(self, ax=None, **matplotlib_kwargs):
        """Plot the empirical distribution function

        Available only if ``matplotlib`` is installed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to draw the plot onto, otherwise uses the current Axes.

        **matplotlib_kwargs : dict, optional
            Keyword arguments passed directly to `matplotlib.axes.Axes.step`.
            Unless overridden, ``where='post'``.

        Returns
        -------
        lines : list of `matplotlib.lines.Line2D`
            Objects representing the plotted data
        """
        try:
            import matplotlib
        except ModuleNotFoundError as exc:
            message = 'matplotlib must be installed to use method `plot`.'
            raise ModuleNotFoundError(message) from exc
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        kwargs = {'where': 'post'}
        kwargs.update(matplotlib_kwargs)
        delta = np.ptp(self.quantiles) * 0.05
        q = self.quantiles
        q = [q[0] - delta] + list(q) + [q[-1] + delta]
        return ax.step(q, self.evaluate(q), **kwargs)

    def confidence_interval(self, confidence_level=0.95, *, method='linear'):
        """Compute a confidence interval around the CDF/SF point estimate

        Parameters
        ----------
        confidence_level : float, default: 0.95
            Confidence level for the computed confidence interval

        method : str, {"linear", "log-log"}
            Method used to compute the confidence interval. Options are
            "linear" for the conventional Greenwood confidence interval
            (default)  and "log-log" for the "exponential Greenwood",
            log-negative-log-transformed confidence interval.

        Returns
        -------
        ci : ``ConfidenceInterval``
            An object with attributes ``low`` and ``high``, instances of
            `~scipy.stats._result_classes.EmpiricalDistributionFunction` that
            represent the lower and upper bounds (respectively) of the
            confidence interval.

        Notes
        -----
        Confidence intervals are computed according to the Greenwood formula
        (``method='linear'``) or the more recent "exponential Greenwood"
        formula (``method='log-log'``) as described in [1]_. The conventional
        Greenwood formula can result in lower confidence limits less than 0
        and upper confidence limits greater than 1; these are clipped to the
        unit interval. NaNs may be produced by either method; these are
        features of the formulas.

        References
        ----------
        .. [1] Sawyer, Stanley. "The Greenwood and Exponential Greenwood
               Confidence Intervals in Survival Analysis."
               https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf

        """
        message = 'Confidence interval bounds do not implement a `confidence_interval` method.'
        if self._n is None:
            raise NotImplementedError(message)
        methods = {'linear': self._linear_ci, 'log-log': self._loglog_ci}
        message = f'`method` must be one of {set(methods)}.'
        if method.lower() not in methods:
            raise ValueError(message)
        message = '`confidence_level` must be a scalar between 0 and 1.'
        confidence_level = np.asarray(confidence_level)[()]
        if confidence_level.shape or not 0 <= confidence_level <= 1:
            raise ValueError(message)
        method_fun = methods[method.lower()]
        low, high = method_fun(confidence_level)
        message = 'The confidence interval is undefined at some observations. This is a feature of the mathematical formula used, not an error in its implementation.'
        if np.any(np.isnan(low) | np.isnan(high)):
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        low, high = (np.clip(low, 0, 1), np.clip(high, 0, 1))
        low = EmpiricalDistributionFunction(self.quantiles, low, None, None, self._kind)
        high = EmpiricalDistributionFunction(self.quantiles, high, None, None, self._kind)
        return ConfidenceInterval(low, high)

    def _linear_ci(self, confidence_level):
        sf, d, n = (self._sf, self._d, self._n)
        with np.errstate(divide='ignore', invalid='ignore'):
            var = sf ** 2 * np.cumsum(d / (n * (n - d)))
        se = np.sqrt(var)
        z = special.ndtri(1 / 2 + confidence_level / 2)
        z_se = z * se
        low = self.probabilities - z_se
        high = self.probabilities + z_se
        return (low, high)

    def _loglog_ci(self, confidence_level):
        sf, d, n = (self._sf, self._d, self._n)
        with np.errstate(divide='ignore', invalid='ignore'):
            var = 1 / np.log(sf) ** 2 * np.cumsum(d / (n * (n - d)))
        se = np.sqrt(var)
        z = special.ndtri(1 / 2 + confidence_level / 2)
        with np.errstate(divide='ignore'):
            lnl_points = np.log(-np.log(sf))
        z_se = z * se
        low = np.exp(-np.exp(lnl_points + z_se))
        high = np.exp(-np.exp(lnl_points - z_se))
        if self._kind == 'cdf':
            low, high = (1 - high, 1 - low)
        return (low, high)