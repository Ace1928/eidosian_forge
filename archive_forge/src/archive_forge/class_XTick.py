import datetime
import functools
import logging
from numbers import Real
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits
class XTick(Tick):
    """
    Contains all the Artists needed to make an x tick - the tick line,
    the label text and the grid line
    """
    __name__ = 'xtick'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ax = self.axes
        self.tick1line.set(data=([0], [0]), transform=ax.get_xaxis_transform('tick1'))
        self.tick2line.set(data=([0], [1]), transform=ax.get_xaxis_transform('tick2'))
        self.gridline.set(data=([0, 0], [0, 1]), transform=ax.get_xaxis_transform('grid'))
        trans, va, ha = self._get_text1_transform()
        self.label1.set(x=0, y=0, verticalalignment=va, horizontalalignment=ha, transform=trans)
        trans, va, ha = self._get_text2_transform()
        self.label2.set(x=0, y=1, verticalalignment=va, horizontalalignment=ha, transform=trans)

    def _get_text1_transform(self):
        return self.axes.get_xaxis_text1_transform(self._pad)

    def _get_text2_transform(self):
        return self.axes.get_xaxis_text2_transform(self._pad)

    def _apply_tickdir(self, tickdir):
        super()._apply_tickdir(tickdir)
        mark1, mark2 = _MARKER_DICT[self._tickdir]
        self.tick1line.set_marker(mark1)
        self.tick2line.set_marker(mark2)

    def update_position(self, loc):
        """Set the location of tick in data coords with scalar *loc*."""
        self.tick1line.set_xdata((loc,))
        self.tick2line.set_xdata((loc,))
        self.gridline.set_xdata((loc,))
        self.label1.set_x(loc)
        self.label2.set_x(loc)
        self._loc = loc
        self.stale = True

    def get_view_interval(self):
        return self.axes.viewLim.intervalx