from contextlib import contextmanager
import datetime
import os
import re
import sys
import types
from collections import deque
from inspect import signature
from io import StringIO
from warnings import warn
from IPython.utils.decorators import undoc
from IPython.utils.py3compat import PYPY
from typing import Dict
def _in_deferred_types(self, cls):
    """
        Check if the given class is specified in the deferred type registry.

        Returns the printer from the registry if it exists, and None if the
        class is not in the registry. Successful matches will be moved to the
        regular type registry for future use.
        """
    mod = _safe_getattr(cls, '__module__', None)
    name = _safe_getattr(cls, '__name__', None)
    key = (mod, name)
    printer = None
    if key in self.deferred_pprinters:
        printer = self.deferred_pprinters.pop(key)
        self.type_pprinters[cls] = printer
    return printer