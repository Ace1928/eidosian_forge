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
def _printf(self, fmt, *args, **print3options):
    """Print to sys.stdout or the configured stream if any is
        specified and if the file keyword argument is not already
        set in the **print3options** for this specific call.
        """
    if self._stream and (not print3options.get('file', None)):
        if args:
            fmt = fmt % args
        _printf(fmt, file=self._stream, **print3options)
    else:
        _printf(fmt, *args, **print3options)