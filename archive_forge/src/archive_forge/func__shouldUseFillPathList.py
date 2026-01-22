from ..Qt import QtCore, QtGui, QtWidgets
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
def _shouldUseFillPathList(self):
    connect = self.opts['connect']
    return isinstance(connect, str) and connect == 'all' and isinstance(self.opts['fillLevel'], (int, float))