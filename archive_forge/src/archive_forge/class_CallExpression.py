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
class CallExpression:
    """ Object which emits a line-wrapped call expression in the form `__name(*args, **kwargs)` """

    def __init__(__self, __name, *args, **kwargs):
        self = __self
        self.name = __name
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def factory(cls, name):

        def inner(*args, **kwargs):
            return cls(name, *args, **kwargs)
        return inner

    def _repr_pretty_(self, p, cycle):
        started = False

        def new_item():
            nonlocal started
            if started:
                p.text(',')
                p.breakable()
            started = True
        prefix = self.name + '('
        with p.group(len(prefix), prefix, ')'):
            for arg in self.args:
                new_item()
                p.pretty(arg)
            for arg_name, arg in self.kwargs.items():
                new_item()
                arg_prefix = arg_name + '='
                with p.group(len(arg_prefix), arg_prefix):
                    p.pretty(arg)