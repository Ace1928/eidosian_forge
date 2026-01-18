import numpy as np
import pandas as pd
from .._utils import resolution
from ..doctools import document
from .stat import stat

    Calculate weighted boxplot plot statistics

    Parameters
    ----------
    x : array_like
        Data
    weights : array_like
        Weights associated with the data.
    whis : float
        Position of the whiskers beyond the interquartile range.
        The data beyond the whisker are considered outliers.

        If a float, the lower whisker is at the lowest datum above
        `Q1 - whis*(Q3-Q1)`, and the upper whisker at the highest
        datum below `Q3 + whis*(Q3-Q1)`, where Q1 and Q3 are the
        first and third quartiles.  The default value of
        `whis = 1.5` corresponds to Tukey's original definition of
        boxplots.

    Notes
    -----
    This method adapted from Matplotlibs boxplot_stats. The key difference
    is the use of a weighted percentile calculation and then using linear
    interpolation to map weight percentiles back to data.
    