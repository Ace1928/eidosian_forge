import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def _set_facecolor(self, c):
    if c is None:
        c = self._get_default_facecolor()
    self._facecolors = mcolors.to_rgba_array(c, self._alpha)
    self.stale = True