import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def _func_fullname(builtin, module, lineno, name):
    if builtin:
        return f'{module}.{name}'
    else:
        return '%s:%d %s' % (module, lineno, name)