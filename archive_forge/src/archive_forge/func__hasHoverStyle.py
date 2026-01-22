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
def _hasHoverStyle(self):
    return any((self.opts['hover' + opt.title()] != _DEFAULT_STYLE[opt] for opt in ['symbol', 'size', 'pen', 'brush']))