import sys
import types as Types
import warnings
import weakref as Weakref
from inspect import isbuiltin, isclass, iscode, isframe, isfunction, ismethod, ismodule
from math import log
from os import curdir, linesep
from struct import calcsize
from gc import get_objects as _getobjects
from gc import get_referents as _getreferents  # containers only?
from array import array as _array  # array type
def _c100(self, stats):
    """Cutoff as percentage (for backward compatibility)"""
    s = int(stats)
    c = int((stats - s) * 100.0 + 0.5) or self.cutoff
    return (s, c)