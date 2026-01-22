import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
class YChildFuncStat(YFuncStat):
    """
    Class holding information for children function stats.
    """
    _KEYS = {'index': 0, 'ncall': 1, 'nactualcall': 2, 'ttot': 3, 'tsub': 4, 'tavg': 5, 'builtin': 6, 'full_name': 7, 'module': 8, 'lineno': 9, 'name': 10}

    def __add__(self, other):
        if other is None:
            return self
        self.nactualcall += other.nactualcall
        self.ncall += other.ncall
        self.ttot += other.ttot
        self.tsub += other.tsub
        self.tavg = self.ttot / self.ncall
        return self