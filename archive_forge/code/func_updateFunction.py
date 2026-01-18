from __future__ import print_function
import gc
import inspect
import os
import sys
import traceback
import types
from .debug import printExc
def updateFunction(old, new, debug, depth=0, visited=None):
    old.__code__ = new.__code__
    old.__defaults__ = new.__defaults__
    if hasattr(old, '__kwdefaults'):
        old.__kwdefaults__ = new.__kwdefaults__
    old.__doc__ = new.__doc__
    if visited is None:
        visited = []
    if old in visited:
        return
    visited.append(old)
    if hasattr(old, '__previous_reload_version__'):
        maxDepth = updateFunction(old.__previous_reload_version__, new, debug, depth=depth + 1, visited=visited)
    else:
        maxDepth = depth
    if depth == 0:
        new.__previous_reload_version__ = old
    return maxDepth