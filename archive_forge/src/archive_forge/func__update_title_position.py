from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
def _update_title_position(self, renderer):
    """
        Update the title position based on the bounding box enclosing
        all the ticklabels and x-axis spine and xlabel...
        """
    if self._autotitlepos is not None and (not self._autotitlepos):
        _log.debug('title position was updated manually, not adjusting')
        return
    titles = (self.title, self._left_title, self._right_title)
    axs = self._twinned_axes.get_siblings(self) + self.child_axes
    for ax in self.child_axes:
        locator = ax.get_axes_locator()
        ax.apply_aspect(locator(self, renderer) if locator else None)
    for title in titles:
        x, _ = title.get_position()
        title.set_position((x, 1.0))
        top = -np.inf
        for ax in axs:
            bb = None
            if ax.xaxis.get_ticks_position() in ['top', 'unknown'] or ax.xaxis.get_label_position() == 'top':
                bb = ax.xaxis.get_tightbbox(renderer)
            if bb is None:
                if 'outline' in ax.spines:
                    bb = ax.spines['outline'].get_window_extent()
                else:
                    bb = ax.get_window_extent(renderer)
            top = max(top, bb.ymax)
            if title.get_text():
                ax.yaxis.get_tightbbox(renderer)
                if ax.yaxis.offsetText.get_text():
                    bb = ax.yaxis.offsetText.get_tightbbox(renderer)
                    if bb.intersection(title.get_tightbbox(renderer), bb):
                        top = bb.ymax
        if top < 0:
            _log.debug('top of Axes not in the figure, so title not moved')
            return
        if title.get_window_extent(renderer).ymin < top:
            _, y = self.transAxes.inverted().transform((0, top))
            title.set_position((x, y))
            if title.get_window_extent(renderer).ymin < top:
                _, y = self.transAxes.inverted().transform((0.0, 2 * top - title.get_window_extent(renderer).ymin))
                title.set_position((x, y))
    ymax = max((title.get_position()[1] for title in titles))
    for title in titles:
        x, _ = title.get_position()
        title.set_position((x, ymax))