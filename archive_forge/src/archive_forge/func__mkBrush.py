import itertools
import math
import weakref
from collections import OrderedDict
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def _mkBrush(*args, **kwargs):
    """
    Wrapper for fn.mkBrush which avoids creating a new QBrush object if passed one as its
    sole argument. This is used to avoid unnecessary cache misses in SymbolAtlas which
    uses the QBrush object id in its key.
    """
    if len(args) == 1 and isinstance(args[0], QtGui.QBrush):
        return args[0]
    else:
        return fn.mkBrush(*args, **kwargs)