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
class _LazyTickList:
    """
    A descriptor for lazy instantiation of tick lists.

    See comment above definition of the ``majorTicks`` and ``minorTicks``
    attributes.
    """

    def __init__(self, major):
        self._major = major

    def __get__(self, instance, owner):
        if instance is None:
            return self
        elif self._major:
            instance.majorTicks = []
            tick = instance._get_tick(major=True)
            instance.majorTicks.append(tick)
            return instance.majorTicks
        else:
            instance.minorTicks = []
            tick = instance._get_tick(major=False)
            instance.minorTicks.append(tick)
            return instance.minorTicks