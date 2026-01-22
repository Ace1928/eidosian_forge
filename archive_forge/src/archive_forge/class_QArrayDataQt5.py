import ctypes
import itertools
import numpy as np
from . import QT_LIB, QtCore, QtGui, compat
class QArrayDataQt5(ctypes.Structure):
    _fields_ = [('ref', ctypes.c_int), ('size', ctypes.c_int), ('alloc', ctypes.c_uint, 31), ('offset', ctypes.c_ssize_t)]