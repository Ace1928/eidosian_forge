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
class EnterWith(Stmt):
    """Enter a "with" context
    """

    def __init__(self, contextmanager, begin, end, loc):
        """
        Parameters
        ----------
        contextmanager : IR value
        begin, end : int
            The beginning and the ending offset of the with-body.
        loc : ir.Loc instance
            Source location
        """
        assert isinstance(contextmanager, Var)
        assert isinstance(loc, Loc)
        self.contextmanager = contextmanager
        self.begin = begin
        self.end = end
        self.loc = loc

    def __str__(self):
        return 'enter_with {}'.format(self.contextmanager)

    def list_vars(self):
        return [self.contextmanager]