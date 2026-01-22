import ctypes
import itertools
import numpy as np
from . import QT_LIB, QtCore, QtGui, compat
class QPainterPathPrivateQt6(ctypes.Structure):
    _fields_ = [('ref', ctypes.c_int), ('adata', ctypes.POINTER(QArrayDataQt6)), ('data', ctypes.c_void_p), ('size', ctypes.c_ssize_t)]