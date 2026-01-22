import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def _get_inverse_paths_linestyles(self):
    """
        Returns the path and pattern for the gaps in the non-solid lines.

        This path and pattern is the inverse of the path and pattern used to
        construct the non-solid lines. For solid lines, we set the inverse path
        to nans to prevent drawing an inverse line.
        """
    path_patterns = [(mpath.Path(np.full((1, 2), np.nan)), ls) if ls == (0, None) else (path, mlines._get_inverse_dash_pattern(*ls)) for path, ls in zip(self._paths, itertools.cycle(self._linestyles))]
    return zip(*path_patterns)