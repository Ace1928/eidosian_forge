import ctypes
import itertools
import numpy as np
from . import QT_LIB, QtCore, QtGui, compat
def _wrap_instances(self, array):
    return list(map(compat.wrapinstance, itertools.count(array.ctypes.data, array.strides[0]), itertools.repeat(self._Klass, array.shape[0])))