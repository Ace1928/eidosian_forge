import decimal
import re
import warnings
from math import isinf, isnan
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
def valueInRange(self, value):
    if not isnan(value):
        bounds = self.opts['bounds']
        if bounds[0] is not None and value < bounds[0]:
            return False
        if bounds[1] is not None and value > bounds[1]:
            return False
        if self.opts.get('int', False):
            if int(value) != value:
                return False
    return True