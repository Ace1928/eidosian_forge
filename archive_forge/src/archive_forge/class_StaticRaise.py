from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import re
import sys
import operator
from types import FunctionType, BuiltinFunctionType
from functools import total_ordering
from io import StringIO
from numba.core import errors, config
from numba.core.utils import (BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS,
from numba.core.errors import (NotDefinedError, RedefinedError,
from numba.core import consts
class StaticRaise(Terminator):
    """
    Raise an exception class and arguments known at compile-time.
    Note that if *exc_class* is None, a bare "raise" statement is implied
    (i.e. re-raise the current exception).
    """
    is_exit = True

    def __init__(self, exc_class, exc_args, loc):
        assert exc_class is None or isinstance(exc_class, type)
        assert isinstance(loc, Loc)
        assert exc_args is None or isinstance(exc_args, tuple)
        self.exc_class = exc_class
        self.exc_args = exc_args
        self.loc = loc

    def __str__(self):
        if self.exc_class is None:
            return '<static> raise'
        elif self.exc_args is None:
            return '<static> raise %s' % (self.exc_class,)
        else:
            return '<static> raise %s(%s)' % (self.exc_class, ', '.join(map(repr, self.exc_args)))

    def get_targets(self):
        return []