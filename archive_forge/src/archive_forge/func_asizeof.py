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
def asizeof(self, *objs, **opts):
    """Return the combined size of the given objects
        (with modified options, see method **set**).
        """
    if opts:
        self.set(**opts)
    self.exclude_refs(*objs)
    return sum((self._sizer(o, 0, 0, None) for o in objs))