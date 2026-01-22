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
class Print(Stmt):
    """
    Print some values.
    """

    def __init__(self, args, vararg, loc):
        assert all((isinstance(x, Var) for x in args))
        assert vararg is None or isinstance(vararg, Var)
        assert isinstance(loc, Loc)
        self.args = tuple(args)
        self.vararg = vararg
        self.consts = {}
        self.loc = loc

    def __str__(self):
        return 'print(%s)' % ', '.join((str(v) for v in self.args))