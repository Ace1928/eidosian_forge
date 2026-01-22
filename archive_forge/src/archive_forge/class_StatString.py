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
class StatString:
    """
    Class to prettify/trim a profile result column.
    """
    _TRAIL_DOT = '..'
    _LEFT = 1
    _RIGHT = 2

    def __init__(self, s):
        self._s = str(s)

    def _trim(self, length, direction):
        if len(self._s) > length:
            if direction == self._LEFT:
                self._s = self._s[-length:]
                return self._TRAIL_DOT + self._s[len(self._TRAIL_DOT):]
            elif direction == self._RIGHT:
                self._s = self._s[:length]
                return self._s[:-len(self._TRAIL_DOT)] + self._TRAIL_DOT
        return self._s + ' ' * (length - len(self._s))

    def ltrim(self, length):
        return self._trim(length, self._LEFT)

    def rtrim(self, length):
        return self._trim(length, self._RIGHT)