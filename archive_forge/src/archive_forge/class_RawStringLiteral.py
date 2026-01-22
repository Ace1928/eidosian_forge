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
class RawStringLiteral:
    """ Wrapper that shows a string with a `r` prefix """

    def __init__(self, value):
        self.value = value

    def _repr_pretty_(self, p, cycle):
        base_repr = repr(self.value)
        if base_repr[:1] in 'uU':
            base_repr = base_repr[1:]
            prefix = 'ur'
        else:
            prefix = 'r'
        base_repr = prefix + base_repr.replace('\\\\', '\\')
        p.text(base_repr)