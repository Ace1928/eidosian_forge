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
def _do_extends(self, ax=None):
    """
        Add the extend tri/rectangles on the outside of the axes.

        ax is unused, but required due to the callbacks on xlim/ylim changed
        """
    for patch in self._extend_patches:
        patch.remove()
    self._extend_patches = []
    _, extendlen = self._proportional_y()
    bot = 0 - (extendlen[0] if self._extend_lower() else 0)
    top = 1 + (extendlen[1] if self._extend_upper() else 0)
    if not self.extendrect:
        xyout = np.array([[0, 0], [0.5, bot], [1, 0], [1, 1], [0.5, top], [0, 1], [0, 0]])
    else:
        xyout = np.array([[0, 0], [0, bot], [1, bot], [1, 0], [1, 1], [1, top], [0, top], [0, 1], [0, 0]])
    if self.orientation == 'horizontal':
        xyout = xyout[:, ::-1]
    self.outline.set_xy(xyout)
    if not self._filled:
        return
    mappable = getattr(self, 'mappable', None)
    if isinstance(mappable, contour.ContourSet) and any((hatch is not None for hatch in mappable.hatches)):
        hatches = mappable.hatches * (len(self._y) + 1)
    else:
        hatches = [None] * (len(self._y) + 1)
    if self._extend_lower():
        if not self.extendrect:
            xy = np.array([[0, 0], [0.5, bot], [1, 0]])
        else:
            xy = np.array([[0, 0], [0, bot], [1.0, bot], [1, 0]])
        if self.orientation == 'horizontal':
            xy = xy[:, ::-1]
        val = -1 if self._long_axis().get_inverted() else 0
        color = self.cmap(self.norm(self._values[val]))
        patch = mpatches.PathPatch(mpath.Path(xy), facecolor=color, alpha=self.alpha, linewidth=0, antialiased=False, transform=self.ax.transAxes, hatch=hatches[0], clip_on=False, zorder=np.nextafter(self.ax.patch.zorder, -np.inf))
        self.ax.add_patch(patch)
        self._extend_patches.append(patch)
        hatches = hatches[1:]
    if self._extend_upper():
        if not self.extendrect:
            xy = np.array([[0, 1], [0.5, top], [1, 1]])
        else:
            xy = np.array([[0, 1], [0, top], [1, top], [1, 1]])
        if self.orientation == 'horizontal':
            xy = xy[:, ::-1]
        val = 0 if self._long_axis().get_inverted() else -1
        color = self.cmap(self.norm(self._values[val]))
        hatch_idx = len(self._y) - 1
        patch = mpatches.PathPatch(mpath.Path(xy), facecolor=color, alpha=self.alpha, linewidth=0, antialiased=False, transform=self.ax.transAxes, hatch=hatches[hatch_idx], clip_on=False, zorder=np.nextafter(self.ax.patch.zorder, -np.inf))
        self.ax.add_patch(patch)
        self._extend_patches.append(patch)
    self._update_dividers()