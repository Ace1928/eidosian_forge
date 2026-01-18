from statsmodels.compat.python import lzip
import numpy as np
from scipy import stats
from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import add_constant
from . import utils

        Plot of unscaled quantiles of x against the prob of a distribution.

        The x-axis is scaled linearly with the quantiles, but the probabilities
        are used to label the axis.

        Parameters
        ----------
        xlabel : {None, str}, optional
            User-provided labels for the x-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        ylabel : {None, str}, optional
            User-provided labels for the y-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        line : {None, "45", "s", "r", q"}, optional
            Options for the reference line to which the data is compared:

            - "45" - 45-degree line
            - "s" - standardized line, the expected order statistics are scaled
              by the standard deviation of the given sample and have the mean
              added to them
            - "r" - A regression line is fit
            - "q" - A line is fit through the quartiles.
            - None - by default no reference line is added to the plot.

        exceed : bool, optional
            If False (default) the raw sample quantiles are plotted against
            the theoretical quantiles, show the probability that a sample will
            not exceed a given value. If True, the theoretical quantiles are
            flipped such that the figure displays the probability that a
            sample will exceed a given value.
        ax : AxesSubplot, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.
        **plotkwargs
            Additional arguments to be passed to the `plot` command.

        Returns
        -------
        Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        