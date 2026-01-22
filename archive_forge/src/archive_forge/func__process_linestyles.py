from contextlib import ExitStack
import functools
import math
from numbers import Integral
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
def _process_linestyles(self, linestyles):
    Nlev = len(self.levels)
    if linestyles is None:
        tlinestyles = ['solid'] * Nlev
        if self.monochrome:
            eps = -(self.zmax - self.zmin) * 1e-15
            for i, lev in enumerate(self.levels):
                if lev < eps:
                    tlinestyles[i] = self.negative_linestyles
    elif isinstance(linestyles, str):
        tlinestyles = [linestyles] * Nlev
    elif np.iterable(linestyles):
        tlinestyles = list(linestyles)
        if len(tlinestyles) < Nlev:
            nreps = int(np.ceil(Nlev / len(linestyles)))
            tlinestyles = tlinestyles * nreps
        if len(tlinestyles) > Nlev:
            tlinestyles = tlinestyles[:Nlev]
    else:
        raise ValueError('Unrecognized type for linestyles kwarg')
    return tlinestyles