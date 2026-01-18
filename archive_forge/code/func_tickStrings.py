import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def tickStrings(self, values, scale, spacing):
    """Return the strings that should be placed next to ticks. This method is called
        when redrawing the axis and is a good method to override in subclasses.
        The method is called with a list of tick values, a scaling factor (see below), and the
        spacing between ticks (this is required since, in some instances, there may be only
        one tick and thus no other way to determine the tick spacing)

        The scale argument is used when the axis label is displaying units which may have an SI scaling prefix.
        When determining the text to display, use value*scale to correctly account for this prefix.
        For example, if the axis label's units are set to 'V', then a tick value of 0.001 might
        be accompanied by a scale value of 1000. This indicates that the label is displaying 'mV', and
        thus the tick should display 0.001 * 1000 = 1.
        """
    if self.logMode:
        return self.logTickStrings(values, scale, spacing)
    places = max(0, ceil(-log10(spacing * scale)))
    strings = []
    for v in values:
        vs = v * scale
        if abs(vs) < 0.001 or abs(vs) >= 10000:
            vstr = '%g' % vs
        else:
            vstr = '%%0.%df' % places % vs
        strings.append(vstr)
    return strings