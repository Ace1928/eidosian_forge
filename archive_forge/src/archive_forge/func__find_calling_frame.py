import logging
import functools
import inspect
import itertools
import sys
import textwrap
import types
from pyomo.common.errors import DeveloperError
def _find_calling_frame(module_offset):
    g = [globals()]
    calling_frame = inspect.currentframe().f_back
    while calling_frame is not None:
        if calling_frame.f_globals is g[-1]:
            calling_frame = calling_frame.f_back
        elif len(g) < module_offset:
            g.append(calling_frame.f_globals)
        else:
            break
    return calling_frame