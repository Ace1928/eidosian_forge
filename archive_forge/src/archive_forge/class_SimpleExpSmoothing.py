from statsmodels.compat.pandas import deprecate_kwarg
import contextlib
from typing import Any
from collections.abc import Hashable, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, least_squares, minimize
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.tools.validation import (
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.exponential_smoothing.ets import (
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import HoltWintersArgs
from statsmodels.tsa.holtwinters._smoothers import (
from statsmodels.tsa.holtwinters.results import (
from statsmodels.tsa.tsatools import freq_to_period
class SimpleExpSmoothing(ExponentialSmoothing):
    """
    Simple Exponential Smoothing

    Parameters
    ----------
    endog : array_like
        The time series to model.
    initialization_method : str, optional
        Method for initialize the recursions. One of:

        * None
        * 'estimated'
        * 'heuristic'
        * 'legacy-heuristic'
        * 'known'

        None defaults to the pre-0.12 behavior where initial values
        are passed as part of ``fit``. If any of the other values are
        passed, then the initial values must also be set when constructing
        the model. If 'known' initialization is used, then `initial_level`
        must be passed, as well as `initial_trend` and `initial_seasonal` if
        applicable. Default is 'estimated'. "legacy-heuristic" uses the same
        values that were used in statsmodels 0.11 and earlier.
    initial_level : float, optional
        The initial level component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.

    See Also
    --------
    ExponentialSmoothing
        Exponential smoothing with trend and seasonal components.
    Holt
        Exponential smoothing with a trend component.

    Notes
    -----
    This is a full implementation of the simple exponential smoothing as
    per [1]_.  `SimpleExpSmoothing` is a restricted version of
    :class:`ExponentialSmoothing`.

    See the notebook `Exponential Smoothing
    <../examples/notebooks/generated/exponential_smoothing.html>`__
    for an overview.

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    def __init__(self, endog, initialization_method=None, initial_level=None):
        super().__init__(endog, initialization_method=initialization_method, initial_level=initial_level)

    def fit(self, smoothing_level=None, *, optimized=True, start_params=None, initial_level=None, use_brute=True, use_boxcox=None, remove_bias=False, method=None, minimize_kwargs=None):
        """
        Fit the model

        Parameters
        ----------
        smoothing_level : float, optional
            The smoothing_level value of the simple exponential smoothing, if
            the value is set then this value will be used as the value.
        optimized : bool, optional
            Estimate model parameters by maximizing the log-likelihood.
        start_params : ndarray, optional
            Starting values to used when optimizing the fit.  If not provided,
            starting values are determined using a combination of grid search
            and reasonable values based on the initial values of the data.
        initial_level : float, optional
            Value to use when initializing the fitted level.
        use_brute : bool, optional
            Search for good starting values using a brute force (grid)
            optimizer. If False, a naive set of starting values is used.
        use_boxcox : {True, False, 'log', float}, optional
            Should the Box-Cox transform be applied to the data first? If 'log'
            then apply the log. If float then use the value as lambda.
        remove_bias : bool, optional
            Remove bias from forecast values and fitted values by enforcing
            that the average residual is equal to zero.
        method : str, default "L-BFGS-B"
            The minimizer used. Valid options are "L-BFGS-B" (default), "TNC",
            "SLSQP", "Powell", "trust-constr", "basinhopping" (also "bh") and
            "least_squares" (also "ls"). basinhopping tries multiple starting
            values in an attempt to find a global minimizer in non-convex
            problems, and so is slower than the others.
        minimize_kwargs : dict[str, Any]
            A dictionary of keyword arguments passed to SciPy's minimize
            function if method is one of "L-BFGS-B" (default), "TNC",
            "SLSQP", "Powell", or "trust-constr", or SciPy's basinhopping
            or least_squares. The valid keywords are optimizer specific.
            Consult SciPy's documentation for the full set of options.

        Returns
        -------
        HoltWintersResults
            See statsmodels.tsa.holtwinters.HoltWintersResults.

        Notes
        -----
        This is a full implementation of the simple exponential smoothing as
        per [1].

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
            and practice. OTexts, 2014.
        """
        return super().fit(smoothing_level=smoothing_level, optimized=optimized, start_params=start_params, initial_level=initial_level, use_brute=use_brute, remove_bias=remove_bias, use_boxcox=use_boxcox, method=method, minimize_kwargs=minimize_kwargs)