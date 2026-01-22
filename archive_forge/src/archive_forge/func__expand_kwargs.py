import warnings
import numpy as np
import pandas as pd
from pandas.plotting import PlotAccessor
from pandas import CategoricalDtype
import geopandas
from packaging.version import Version
from ._decorator import doc
def _expand_kwargs(kwargs, multiindex):
    """
    Most arguments to the plot functions must be a (single) value, or a sequence
    of values. This function checks each key-value pair in 'kwargs' and expands
    it (in place) to the correct length/formats with help of 'multiindex', unless
    the value appears to already be a valid (single) value for the key.
    """
    import matplotlib
    from matplotlib.colors import is_color_like
    from typing import Iterable
    mpl = Version(matplotlib.__version__)
    if mpl >= Version('3.4'):
        scalar_kwargs = ['marker', 'path_effects']
    else:
        scalar_kwargs = ['marker', 'alpha', 'path_effects']
    for att, value in kwargs.items():
        if 'color' in att:
            if is_color_like(value):
                continue
        elif 'linestyle' in att:
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Iterable):
                continue
        elif att in scalar_kwargs:
            continue
        if pd.api.types.is_list_like(value):
            kwargs[att] = np.take(value, multiindex, axis=0)