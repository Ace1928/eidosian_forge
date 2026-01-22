import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
class ColorShell(object):
    _has_initialized = False

    def __init__(self):
        init()
        self._has_initialized = True

    def __enter__(self):
        if self._has_initialized:
            reinit()

    def __exit__(self, *exc_detail):
        Style.RESET_ALL
        deinit()