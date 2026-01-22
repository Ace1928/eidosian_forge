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
def _in_doctest(self):
    return len(self.scopeStack) >= 2 and isinstance(self.scopeStack[1], DoctestScope)