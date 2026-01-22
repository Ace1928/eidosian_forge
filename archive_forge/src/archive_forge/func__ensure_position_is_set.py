from collections.abc import MutableMapping
import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.artist import allow_rasterization
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.path as mpath
def _ensure_position_is_set(self):
    if self._position is None:
        self._position = ('outward', 0.0)
        self.set_position(self._position)