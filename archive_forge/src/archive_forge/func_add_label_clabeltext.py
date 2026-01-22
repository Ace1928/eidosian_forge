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
def add_label_clabeltext(self, x, y, rotation, lev, cvalue):
    """Add contour label with `.Text.set_transform_rotates_text`."""
    self.add_label(x, y, rotation, lev, cvalue)
    t = self.labelTexts[-1]
    data_rotation, = self.axes.transData.inverted().transform_angles([rotation], [[x, y]])
    t.set(rotation=data_rotation, transform_rotates_text=True)