import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def _unknown_handler(self, node):
    if os.environ.get('PYFLAKES_ERROR_UNKNOWN'):
        raise NotImplementedError(f'Unexpected type: {type(node)}')
    else:
        self.handleChildren(node)