import os
import sys
import tempfile
import operator
import functools
import itertools
import re
import contextlib
import pickle
import textwrap
import builtins
import pkg_resources
from distutils.errors import DistutilsError
from pkg_resources import working_set
class ExceptionSaver:
    """
    A Context Manager that will save an exception, serialize, and restore it
    later.
    """

    def __enter__(self):
        return self

    def __exit__(self, type, exc, tb):
        if not exc:
            return False
        self._saved = UnpickleableException.dump(type, exc)
        self._tb = tb
        return True

    def resume(self):
        """restore and re-raise any exception"""
        if '_saved' not in vars(self):
            return
        type, exc = map(pickle.loads, self._saved)
        raise exc.with_traceback(self._tb)