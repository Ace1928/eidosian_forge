import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
def _preprocess_colors(self, data, colors, axis):
    """Preprocess {row/col}_colors to extract labels and convert colors."""
    labels = None
    if colors is not None:
        if isinstance(colors, (pd.DataFrame, pd.Series)):
            if not hasattr(data, 'index') and axis == 0 or (not hasattr(data, 'columns') and axis == 1):
                axis_name = 'col' if axis else 'row'
                msg = f"{axis_name}_colors indices can't be matched with data indices. Provide {axis_name}_colors as a non-indexed datatype, e.g. by using `.to_numpy()``"
                raise TypeError(msg)
            if axis == 0:
                colors = colors.reindex(data.index)
            else:
                colors = colors.reindex(data.columns)
            colors = colors.astype(object).fillna('white')
            if isinstance(colors, pd.DataFrame):
                labels = list(colors.columns)
                colors = colors.T.values
            else:
                if colors.name is None:
                    labels = ['']
                else:
                    labels = [colors.name]
                colors = colors.values
        colors = _convert_colors(colors)
    return (colors, labels)