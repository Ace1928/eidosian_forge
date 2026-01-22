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
class PrintDetector(object):
    """Find code locations that print to stdout."""

    def __init__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def remove(self):
        sys.stdout = self.stdout

    def __del__(self):
        self.remove()

    def write(self, x):
        self.stdout.write(x)
        traceback.print_stack()

    def flush(self):
        self.stdout.flush()