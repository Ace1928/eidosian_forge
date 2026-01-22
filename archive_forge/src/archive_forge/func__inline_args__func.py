import os
from io import open
import types
from functools import wraps, partial
from contextlib import contextmanager
import sys, re
import sre_parse
import sre_constants
from inspect import getmembers, getmro
from functools import partial, wraps
from itertools import repeat, product
def _inline_args__func(func):

    @wraps(func)
    def create_decorator(_f, with_self):
        if with_self:

            def f(self, children):
                return _f(self, *children)
        else:

            def f(self, children):
                return _f(*children)
        return f
    return smart_decorator(func, create_decorator)