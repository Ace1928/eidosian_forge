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
def _find_lines_from_code(code, strs):
    """Return dict where keys are lines in the line number table."""
    linenos = {}
    for _, lineno in dis.findlinestarts(code):
        if lineno not in strs:
            linenos[lineno] = 1
    return linenos