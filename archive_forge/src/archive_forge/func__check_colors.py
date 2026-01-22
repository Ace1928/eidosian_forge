from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import Series
import pandas._testing as tm
def _check_colors(collections, linecolors=None, facecolors=None, mapping=None):
    """
    Check each artist has expected line colors and face colors

    Parameters
    ----------
    collections : list-like
        list or collection of target artist
    linecolors : list-like which has the same length as collections
        list of expected line colors
    facecolors : list-like which has the same length as collections
        list of expected face colors
    mapping : Series
        Series used for color grouping key
        used for andrew_curves, parallel_coordinates, radviz test
    """
    from matplotlib import colors
    from matplotlib.collections import Collection, LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    conv = colors.ColorConverter
    if linecolors is not None:
        if mapping is not None:
            linecolors = _get_colors_mapped(mapping, linecolors)
            linecolors = linecolors[:len(collections)]
        assert len(collections) == len(linecolors)
        for patch, color in zip(collections, linecolors):
            if isinstance(patch, Line2D):
                result = patch.get_color()
                result = conv.to_rgba(result)
            elif isinstance(patch, (PolyCollection, LineCollection)):
                result = tuple(patch.get_edgecolor()[0])
            else:
                result = patch.get_edgecolor()
            expected = conv.to_rgba(color)
            assert result == expected
    if facecolors is not None:
        if mapping is not None:
            facecolors = _get_colors_mapped(mapping, facecolors)
            facecolors = facecolors[:len(collections)]
        assert len(collections) == len(facecolors)
        for patch, color in zip(collections, facecolors):
            if isinstance(patch, Collection):
                result = patch.get_facecolor()[0]
            else:
                result = patch.get_facecolor()
            if isinstance(result, np.ndarray):
                result = tuple(result)
            expected = conv.to_rgba(color)
            assert result == expected