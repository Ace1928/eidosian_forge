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
def _file(self, path, mode='r', *args, **kw):
    if mode not in ('r', 'rt', 'rb', 'rU', 'U') and (not self._ok(path)):
        self._violation('file', path, mode, *args, **kw)
    return _file(path, mode, *args, **kw)