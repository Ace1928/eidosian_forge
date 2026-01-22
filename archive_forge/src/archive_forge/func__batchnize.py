from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def _batchnize(classname, name):
    """Enables batch request for the method classname.name."""
    func = getattr(classname, name, None)

    def _func(v, n):
        if type(n) is int and (n < 0 or n >= v.piece_size()):
            raise IndexError('piece id is out of range.')
        return func(v, n)

    def _batched_func(self, arg):
        if type(arg) is list:
            return [_func(self, n) for n in arg]
        else:
            return _func(self, arg)
    setattr(classname, name, _batched_func)