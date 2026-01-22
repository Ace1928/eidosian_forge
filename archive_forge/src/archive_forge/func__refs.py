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
def _refs(obj, named, *attrs, **kwds):
    """Return specific attribute objects of an object."""
    if named:
        _N = _NamedRef
    else:

        def _N(unused, o):
            return o
    for a in attrs:
        if hasattr(obj, a):
            yield _N(a, getattr(obj, a))
    if kwds:
        for a, o in _dir2(obj, **kwds):
            yield _N(a, o)