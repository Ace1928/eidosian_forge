import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
class AutoMinorLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks. The scale must be linear with major ticks evenly spaced.
    """

    def __init__(self, n=None):
        """
        *n* is the number of subdivisions of the interval between
        major ticks; e.g., n=2 will place a single minor tick midway
        between major ticks.

        If *n* is omitted or None, the value stored in rcParams will be used.
        In case *n* is set to 'auto', it will be set to 4 or 5. If the distance
        between the major ticks equals 1, 2.5, 5 or 10 it can be perfectly
        divided in 5 equidistant sub-intervals with a length multiple of
        0.05. Otherwise it is divided in 4 sub-intervals.
        """
        self.ndivs = n

    def __call__(self):
        """Return the locations of the ticks."""
        if self.axis.get_scale() == 'log':
            _api.warn_external('AutoMinorLocator does not work with logarithmic scale')
            return []
        majorlocs = np.unique(self.axis.get_majorticklocs())
        try:
            majorstep = majorlocs[1] - majorlocs[0]
        except IndexError:
            return []
        if self.ndivs is None:
            if self.axis.axis_name == 'y':
                self.ndivs = mpl.rcParams['ytick.minor.ndivs']
            else:
                self.ndivs = mpl.rcParams['xtick.minor.ndivs']
        if self.ndivs == 'auto':
            majorstep_no_exponent = 10 ** (np.log10(majorstep) % 1)
            if np.isclose(majorstep_no_exponent, [1.0, 2.5, 5.0, 10.0]).any():
                ndivs = 5
            else:
                ndivs = 4
        else:
            ndivs = self.ndivs
        minorstep = majorstep / ndivs
        vmin, vmax = self.axis.get_view_interval()
        if vmin > vmax:
            vmin, vmax = (vmax, vmin)
        t0 = majorlocs[0]
        tmin = round((vmin - t0) / minorstep)
        tmax = round((vmax - t0) / minorstep) + 1
        locs = np.arange(tmin, tmax) * minorstep + t0
        return self.raise_if_exceeds(locs)

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a %s type.' % type(self))