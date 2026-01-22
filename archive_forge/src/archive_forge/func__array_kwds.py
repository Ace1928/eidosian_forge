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
def _array_kwds(obj):
    b = max(56, _getsizeof(obj, 0) - _len_array(obj))
    return dict(base=b, leng=_len_array, item=_sizeof_Cbyte, vari='itemsize', xtyp=True)