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
def add_label_near(self, x, y, inline=True, inline_spacing=5, transform=None):
    """
        Add a label near the point ``(x, y)``.

        Parameters
        ----------
        x, y : float
            The approximate location of the label.
        inline : bool, default: True
            If *True* remove the segment of the contour beneath the label.
        inline_spacing : int, default: 5
            Space in pixels to leave on each side of label when placing
            inline. This spacing will be exact for labels at locations where
            the contour is straight, less so for labels on curved contours.
        transform : `.Transform` or `False`, default: ``self.axes.transData``
            A transform applied to ``(x, y)`` before labeling.  The default
            causes ``(x, y)`` to be interpreted as data coordinates.  `False`
            is a synonym for `.IdentityTransform`; i.e. ``(x, y)`` should be
            interpreted as display coordinates.
        """
    if transform is None:
        transform = self.axes.transData
    if transform:
        x, y = transform.transform((x, y))
    idx_level_min, idx_vtx_min, proj = self._find_nearest_contour((x, y), self.labelIndiceList)
    path = self._paths[idx_level_min]
    level = self.labelIndiceList.index(idx_level_min)
    label_width = self._get_nth_label_width(level)
    rotation, path = self._split_path_and_get_label_rotation(path, idx_vtx_min, proj, label_width, inline_spacing)
    self.add_label(*proj, rotation, self.labelLevelList[idx_level_min], self.labelCValueList[idx_level_min])
    if inline:
        self._paths[idx_level_min] = path