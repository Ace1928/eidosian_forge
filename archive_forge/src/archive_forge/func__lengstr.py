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
def _lengstr(obj):
    """Object length as a string."""
    n = leng(obj)
    if n is None:
        r = _NN
    else:
        x = '!' if n > _len(obj) else _NN
        r = ' leng %d%s' % (n, x)
    return r