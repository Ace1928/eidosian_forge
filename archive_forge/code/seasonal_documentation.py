import numpy as np
import pandas as pd
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tools.validation import PandasWrapper, array_like
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.stl.mstl import MSTL
from statsmodels.tsa.tsatools import freq_to_period

        Plot estimated components

        Parameters
        ----------
        observed : bool
            Include the observed series in the plot
        seasonal : bool
            Include the seasonal component in the plot
        trend : bool
            Include the trend component in the plot
        resid : bool
            Include the residual in the plot
        weights : bool
            Include the weights in the plot (if any)

        Returns
        -------
        matplotlib.figure.Figure
            The figure instance that containing the plot.
        