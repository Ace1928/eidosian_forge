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
class Yield(Inst):

    def __init__(self, value, loc, index):
        assert isinstance(value, Var)
        assert isinstance(loc, Loc)
        self.value = value
        self.loc = loc
        self.index = index

    def __str__(self):
        return 'yield %s' % (self.value,)

    def list_vars(self):
        return [self.value]