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
def _draw_all(self):
    """
        Calculate any free parameters based on the current cmap and norm,
        and do all the drawing.
        """
    if self.orientation == 'vertical':
        if mpl.rcParams['ytick.minor.visible']:
            self.minorticks_on()
    elif mpl.rcParams['xtick.minor.visible']:
        self.minorticks_on()
    self._long_axis().set(label_position=self.ticklocation, ticks_position=self.ticklocation)
    self._short_axis().set_ticks([])
    self._short_axis().set_ticks([], minor=True)
    self._process_values()
    self.vmin, self.vmax = self._boundaries[self._inside][[0, -1]]
    X, Y = self._mesh()
    self._do_extends()
    lower, upper = (self.vmin, self.vmax)
    if self._long_axis().get_inverted():
        lower, upper = (upper, lower)
    if self.orientation == 'vertical':
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(lower, upper)
    else:
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(lower, upper)
    self.update_ticks()
    if self._filled:
        ind = np.arange(len(self._values))
        if self._extend_lower():
            ind = ind[1:]
        if self._extend_upper():
            ind = ind[:-1]
        self._add_solids(X, Y, self._values[ind, np.newaxis])