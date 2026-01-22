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
def _typedef_both(t, base=0, item=0, leng=None, refs=None, kind=_kind_static, heap=False, vari=_Not_vari):
    """Add new typedef for both data and code."""
    v = _Typedef(base=_basicsize(t, base=base), item=_itemsize(t, item), refs=refs, leng=leng, both=True, kind=kind, type=t, vari=vari)
    v.save(t, base=base, heap=heap)
    return v