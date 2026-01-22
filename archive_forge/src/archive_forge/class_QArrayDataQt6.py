import ctypes
import itertools
import numpy as np
from . import QT_LIB, QtCore, QtGui, compat
class QArrayDataQt6(ctypes.Structure):
    _fields_ = [('ref', ctypes.c_int), ('flags', ctypes.c_uint), ('alloc', ctypes.c_ssize_t)]