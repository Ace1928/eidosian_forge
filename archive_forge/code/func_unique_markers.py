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
def unique_markers(n):
    """Build an arbitrarily long list of unique marker styles for points.

    Parameters
    ----------
    n : int
        Number of unique marker specs to generate.

    Returns
    -------
    markers : list of string or tuples
        Values for defining :class:`matplotlib.markers.MarkerStyle` objects.
        All markers will be filled.

    """
    markers = ['o', 'X', (4, 0, 45), 'P', (4, 0, 0), (4, 1, 0), '^', (4, 1, 45), 'v']
    s = 5
    while len(markers) < n:
        a = 360 / (s + 1) / 2
        markers.extend([(s + 1, 1, a), (s + 1, 0, a), (s, 1, 0), (s, 0, 0)])
        s += 1
    return markers[:n]