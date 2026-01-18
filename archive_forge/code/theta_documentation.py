from typing import TYPE_CHECKING, Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.validation import (
from statsmodels.tsa.deterministic import DeterministicTerm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.exponential_smoothing import (
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.tsatools import add_trend, freq_to_period

        Plot forecasts, prediction intervals and in-sample values

        Parameters
        ----------
        steps : int, default 1
            The number of steps ahead to compute the forecast components.
        theta : float, default 2
            The theta value to use when computing the weight to combine
            the trend and the SES forecasts.
        alpha : {float, None}, default 0.05
            The tail probability not covered by the confidence interval. Must
            be in (0, 1). Confidence interval is constructed assuming normally
            distributed shocks. If None, figure will not show the confidence
            interval.
        in_sample : bool, default False
            Flag indicating whether to include the in-sample period in the
            plot.
        fig : Figure, default None
            An existing figure handle. If not provided, a new figure is
            created.
        figsize: tuple[float, float], default None
            Tuple containing the figure size.

        Returns
        -------
        Figure
            Figure handle containing the plot.

        Notes
        -----
        The variance of the h-step forecast is assumed to follow from the
        integrated Moving Average structure of the Theta model, and so is
        :math:`\sigma^2(\alpha^2 + (h-1))`. The prediction interval assumes
        that innovations are normally distributed.
        