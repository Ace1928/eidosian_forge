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
def _OptionError(where, Error=ValueError, **options):
    """Format an *Error* instance for invalid *option* or *options*."""
    t = (_plural(len(options)), _nameof(where), _kwdstr(**options))
    return Error('invalid option%s: %s(%s)' % t)