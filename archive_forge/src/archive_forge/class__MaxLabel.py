import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
class _MaxLabel:

    def __init__(self, value=0):
        self._value = value

    def next(self):
        self._value += 1
        return self._value

    def update(self, newval):
        self._value = max(newval, self._value)