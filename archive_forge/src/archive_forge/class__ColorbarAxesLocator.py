import logging
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import _docstring
class _ColorbarAxesLocator:
    """
    Shrink the axes if there are triangular or rectangular extends.
    """

    def __init__(self, cbar):
        self._cbar = cbar
        self._orig_locator = cbar.ax._axes_locator

    def __call__(self, ax, renderer):
        if self._orig_locator is not None:
            pos = self._orig_locator(ax, renderer)
        else:
            pos = ax.get_position(original=True)
        if self._cbar.extend == 'neither':
            return pos
        y, extendlen = self._cbar._proportional_y()
        if not self._cbar._extend_lower():
            extendlen[0] = 0
        if not self._cbar._extend_upper():
            extendlen[1] = 0
        len = sum(extendlen) + 1
        shrink = 1 / len
        offset = extendlen[0] / len
        if hasattr(ax, '_colorbar_info'):
            aspect = ax._colorbar_info['aspect']
        else:
            aspect = False
        if self._cbar.orientation == 'vertical':
            if aspect:
                self._cbar.ax.set_box_aspect(aspect * shrink)
            pos = pos.shrunk(1, shrink).translated(0, offset * pos.height)
        else:
            if aspect:
                self._cbar.ax.set_box_aspect(1 / (aspect * shrink))
            pos = pos.shrunk(shrink, 1).translated(offset * pos.width, 0)
        return pos

    def get_subplotspec(self):
        return self._cbar.ax.get_subplotspec() or getattr(self._orig_locator, 'get_subplotspec', lambda: None)()