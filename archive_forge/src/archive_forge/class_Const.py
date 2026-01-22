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
class Const(EqualityCheckMixin, AbstractRHS):

    def __init__(self, value, loc, use_literal_type=True):
        assert isinstance(loc, Loc)
        self.value = value
        self.loc = loc
        self.use_literal_type = use_literal_type

    def __repr__(self):
        return 'const(%s, %s)' % (type(self.value).__name__, self.value)

    def infer_constant(self):
        return self.value

    def __deepcopy__(self, memo):
        return Const(value=self.value, loc=self.loc, use_literal_type=self.use_literal_type)