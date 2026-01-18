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
def update_normal(self, mappable):
    """
        Update solid patches, lines, etc.

        This is meant to be called when the norm of the image or contour plot
        to which this colorbar belongs changes.

        If the norm on the mappable is different than before, this resets the
        locator and formatter for the axis, so if these have been customized,
        they will need to be customized again.  However, if the norm only
        changes values of *vmin*, *vmax* or *cmap* then the old formatter
        and locator will be preserved.
        """
    _log.debug('colorbar update normal %r %r', mappable.norm, self.norm)
    self.mappable = mappable
    self.set_alpha(mappable.get_alpha())
    self.cmap = mappable.cmap
    if mappable.norm != self.norm:
        self.norm = mappable.norm
        self._reset_locator_formatter_scale()
    self._draw_all()
    if isinstance(self.mappable, contour.ContourSet):
        CS = self.mappable
        if not CS.filled:
            self.add_lines(CS)
    self.stale = True