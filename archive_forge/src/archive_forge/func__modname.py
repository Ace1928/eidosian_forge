import io
import linecache
import os
import sys
import sysconfig
import token
import tokenize
import inspect
import gc
import dis
import pickle
from time import monotonic as _time
import threading
def _modname(path):
    """Return a plausible module name for the path."""
    base = os.path.basename(path)
    filename, ext = os.path.splitext(base)
    return filename