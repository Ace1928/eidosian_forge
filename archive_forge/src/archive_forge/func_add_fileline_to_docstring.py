import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def add_fileline_to_docstring(module, incursive=True):
    """Append the definition position to each function contained in module.

    Examples
    --------
    # Put the following codes at the end of a file
    add_fileline_to_docstring(__name__)
    """

    def _add_fileline(obj):
        """Add fileinto to a object.
        """
        if obj.__doc__ is None or 'From:' in obj.__doc__:
            return
        fname = inspect.getsourcefile(obj)
        if fname is None:
            return
        try:
            line = inspect.getsourcelines(obj)[-1]
        except IOError:
            return
        obj.__doc__ += '\n\nFrom:%s:%d' % (fname, line)
    if isinstance(module, str):
        module = sys.modules[module]
    for _, obj in inspect.getmembers(module):
        if inspect.isbuiltin(obj):
            continue
        if inspect.isfunction(obj):
            _add_fileline(obj)
        if inspect.ismethod(obj):
            _add_fileline(obj.__func__)
        if inspect.isclass(obj) and incursive:
            add_fileline_to_docstring(obj, False)