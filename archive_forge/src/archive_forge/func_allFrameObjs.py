from __future__ import print_function
import contextlib
import cProfile
import gc
import inspect
import os
import re
import sys
import threading
import time
import traceback
import types
import warnings
import weakref
from time import perf_counter
from numpy import ndarray
from .Qt import QT_LIB, QtCore
from .util import cprint
from .util.mutex import Mutex
def allFrameObjs():
    """Return list of frame objects in current stack. Useful if you want to ignore these objects in refernece searches"""
    f = sys._getframe()
    objs = []
    while f is not None:
        objs.append(f)
        objs.append(f.f_code)
        f = f.f_back
    return objs