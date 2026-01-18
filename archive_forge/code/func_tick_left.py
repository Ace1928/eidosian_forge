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
def tick_left(self):
    """
        Move ticks and ticklabels (if present) to the left of the Axes.
        """
    label = True
    if 'label1On' in self._major_tick_kw:
        label = self._major_tick_kw['label1On'] or self._major_tick_kw['label2On']
    self.set_ticks_position('left')
    self.set_tick_params(which='both', labelleft=label)