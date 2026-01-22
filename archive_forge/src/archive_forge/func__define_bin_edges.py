from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import pandas as pd
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._stats.base import Stat
from typing import TYPE_CHECKING
def _define_bin_edges(self, vals, weight, bins, binwidth, binrange, discrete):
    """Inner function that takes bin parameters as arguments."""
    vals = vals.replace(-np.inf, np.nan).replace(np.inf, np.nan).dropna()
    if binrange is None:
        start, stop = (vals.min(), vals.max())
    else:
        start, stop = binrange
    if discrete:
        bin_edges = np.arange(start - 0.5, stop + 1.5)
    else:
        if binwidth is not None:
            bins = int(round((stop - start) / binwidth))
        bin_edges = np.histogram_bin_edges(vals, bins, binrange, weight)
    return bin_edges