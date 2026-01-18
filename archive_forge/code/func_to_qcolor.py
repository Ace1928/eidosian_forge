from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
def to_qcolor(color):
    """Create a QColor from a matplotlib color"""
    qcolor = QtGui.QColor()
    try:
        rgba = mcolors.to_rgba(color)
    except ValueError:
        _api.warn_external(f'Ignoring invalid color {color!r}')
        return qcolor
    qcolor.setRgbF(*rgba)
    return qcolor