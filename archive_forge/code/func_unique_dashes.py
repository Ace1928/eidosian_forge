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
def unique_dashes(n):
    """Build an arbitrarily long list of unique dash styles for lines.

    Parameters
    ----------
    n : int
        Number of unique dash specs to generate.

    Returns
    -------
    dashes : list of strings or tuples
        Valid arguments for the ``dashes`` parameter on
        :class:`matplotlib.lines.Line2D`. The first spec is a solid
        line (``""``), the remainder are sequences of long and short
        dashes.

    """
    dashes = ['', (4, 1.5), (1, 1), (3, 1.25, 1.5, 1.25), (5, 1, 1, 1)]
    p = 3
    while len(dashes) < n:
        a = itertools.combinations_with_replacement([3, 1.25], p)
        b = itertools.combinations_with_replacement([4, 1], p)
        segment_list = itertools.chain(*zip(list(a)[1:-1][::-1], list(b)[1:-1]))
        for segments in segment_list:
            gap = min(segments)
            spec = tuple(itertools.chain(*((seg, gap) for seg in segments)))
            dashes.append(spec)
        p += 1
    return dashes[:n]