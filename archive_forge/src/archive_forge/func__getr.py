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
def _getr(slist, olist, first=True):
    i = 0
    for e in slist:
        oid = id(e)
        typ = type(e)
        if oid in olist or typ is int:
            continue
        olist[oid] = e
        if first and i % 1000 == 0:
            gc.collect()
        tl = gc.get_referents(e)
        if tl:
            _getr(tl, olist, first=False)
        i += 1