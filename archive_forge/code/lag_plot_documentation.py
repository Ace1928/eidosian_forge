import numpy as np
import pandas as pd
from ..util import with_hv_extension
from .core import hvPlotTabular
Lag plot for time series.

    Parameters:
    -----------
    data: Time series
    lag: lag of the scatter plot, default 1
    kwds: hvplot.scatter options, optional

    Returns:
    --------
    obj : HoloViews object
        The HoloViews representation of the plot.
    