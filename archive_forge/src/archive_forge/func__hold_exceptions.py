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
@contextmanager
def _hold_exceptions(self, exceptions):
    """
            Context manager to ensure proper cleaning of exceptions references
            When given a chained exception instead of a traceback,
            pdb may hold references to many objects which may leak memory.
            We use this context manager to make sure everything is properly cleaned
            """
    try:
        self._chained_exceptions = exceptions
        self._chained_exception_index = len(exceptions) - 1
        yield
    finally:
        self._chained_exceptions = tuple()
        self._chained_exception_index = 0