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
class StaticSetItem(Stmt):
    """
    target[constant index] = value
    """

    def __init__(self, target, index, index_var, value, loc):
        assert isinstance(target, Var)
        assert not isinstance(index, Var)
        assert isinstance(index_var, Var)
        assert isinstance(value, Var)
        assert isinstance(loc, Loc)
        self.target = target
        self.index = index
        self.index_var = index_var
        self.value = value
        self.loc = loc

    def __repr__(self):
        return '%s[%r] = %s' % (self.target, self.index, self.value)