import inspect
import linecache
import sys
import re
import os
from IPython import get_ipython
from contextlib import contextmanager
from IPython.utils import PyColorize
from IPython.utils import coloransi, py3compat
from IPython.core.excolors import exception_colors
from pdb import Pdb as OldPdb
def _is_in_decorator_internal_and_should_skip(self, frame):
    """
        Utility to tell us whether we are in a decorator internal and should stop.

        """
    if not self._predicates['debuggerskip']:
        return False
    if DEBUGGERSKIP in frame.f_code.co_varnames:
        return True
    cframe = frame
    while getattr(cframe, 'f_back', None):
        cframe = cframe.f_back
        if self._get_frame_locals(cframe).get(DEBUGGERSKIP):
            return True
    return False