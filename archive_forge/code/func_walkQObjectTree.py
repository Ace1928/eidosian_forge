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
def walkQObjectTree(obj, counts=None, verbose=False, depth=0):
    """
    Walk through a tree of QObjects, doing nothing to them.
    The purpose of this function is to find dead objects and generate a crash
    immediately rather than stumbling upon them later.
    Prints a count of the objects encountered, for fun. (or is it?)
    """
    if verbose:
        print('  ' * depth + typeStr(obj))
    if counts is None:
        counts = {}
    typ = str(type(obj))
    try:
        counts[typ] += 1
    except KeyError:
        counts[typ] = 1
    for child in obj.children():
        walkQObjectTree(child, counts, verbose, depth + 1)
    return counts