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
def _set_tick_locations(self, ticks, *, minor=False):
    ticks = self.convert_units(ticks)
    locator = mticker.FixedLocator(ticks)
    if len(ticks):
        for axis in self._get_shared_axis():
            axis.set_view_interval(min(ticks), max(ticks))
    self.axes.stale = True
    if minor:
        self.set_minor_locator(locator)
        return self.get_minor_ticks(len(ticks))
    else:
        self.set_major_locator(locator)
        return self.get_major_ticks(len(ticks))