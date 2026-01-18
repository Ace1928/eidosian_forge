import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def tickSpacing(self, minVal, maxVal, size):
    """Return values describing the desired spacing and offset of ticks.

        This method is called whenever the axis needs to be redrawn and is a
        good method to override in subclasses that require control over tick locations.

        The return value must be a list of tuples, one for each set of ticks::

            [
                (major tick spacing, offset),
                (minor tick spacing, offset),
                (sub-minor tick spacing, offset),
                ...
            ]
        """
    if self._tickSpacing is not None:
        return self._tickSpacing
    dif = abs(maxVal - minVal)
    if dif == 0:
        return []
    ref_size = 300.0
    minNumberOfIntervals = max(2.25, 2.25 * self._tickDensity * sqrt(size / ref_size))
    majorMaxSpacing = dif / minNumberOfIntervals
    mantissa, exp2 = frexp(majorMaxSpacing)
    p10unit = 10.0 ** (floor((exp2 - 1) / 3.32192809488736) - 1)
    if 100.0 * p10unit <= majorMaxSpacing:
        majorScaleFactor = 10
        p10unit *= 10.0
    else:
        for majorScaleFactor in (50, 20, 10):
            if majorScaleFactor * p10unit <= majorMaxSpacing:
                break
    majorInterval = majorScaleFactor * p10unit
    minorMinSpacing = 2 * dif / size
    if majorScaleFactor == 10:
        trials = (5, 10)
    else:
        trials = (10, 20, 50)
    for minorScaleFactor in trials:
        minorInterval = minorScaleFactor * p10unit
        if minorInterval >= minorMinSpacing:
            break
    levels = [(majorInterval, 0), (minorInterval, 0)]
    if self.style['maxTickLevel'] >= 2:
        if majorScaleFactor == 10:
            trials = (1, 2, 5, 10)
        elif majorScaleFactor == 20:
            trials = (2, 5, 10, 20)
        elif majorScaleFactor == 50:
            trials = (5, 10, 50)
        else:
            trials = ()
            extraInterval = minorInterval
        for extraScaleFactor in trials:
            extraInterval = extraScaleFactor * p10unit
            if extraInterval >= minorMinSpacing or extraInterval == minorInterval:
                break
        if extraInterval < minorInterval:
            levels.append((extraInterval, 0))
    return levels