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
def BdbQuit_IPython_excepthook(self, et, ev, tb, tb_offset=None):
    raise ValueError('`BdbQuit_IPython_excepthook` is deprecated since version 5.1', DeprecationWarning, stacklevel=2)