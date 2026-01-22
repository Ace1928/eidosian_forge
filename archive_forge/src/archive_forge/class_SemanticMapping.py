from __future__ import annotations
import warnings
import itertools
from copy import copy
from collections import UserString
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from seaborn._core.data import PlotData
from seaborn.palettes import (
from seaborn.utils import (
class SemanticMapping:
    """Base class for mapping data values to plot attributes."""
    map_type: str | None = None
    levels = None
    lookup_table = None

    def __init__(self, plotter):
        self.plotter = plotter

    def _check_list_length(self, levels, values, variable):
        """Input check when values are provided as a list."""
        message = ''
        if len(levels) > len(values):
            message = ' '.join([f'\nThe {variable} list has fewer values ({len(values)})', f'than needed ({len(levels)}) and will cycle, which may', 'produce an uninterpretable plot.'])
            values = [x for _, x in zip(levels, itertools.cycle(values))]
        elif len(values) > len(levels):
            message = ' '.join([f'The {variable} list has more values ({len(values)})', f'than needed ({len(levels)}), which may not be intended.'])
            values = values[:len(levels)]
        if message:
            warnings.warn(message, UserWarning, stacklevel=6)
        return values

    def _lookup_single(self, key):
        """Apply the mapping to a single data value."""
        return self.lookup_table[key]

    def __call__(self, key, *args, **kwargs):
        """Get the attribute(s) values for the data key."""
        if isinstance(key, (list, np.ndarray, pd.Series)):
            return [self._lookup_single(k, *args, **kwargs) for k in key]
        else:
            return self._lookup_single(key, *args, **kwargs)