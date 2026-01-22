import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def _freeze_string(self, string):
    """
        Freeze a Python string object into the code.
        """
    return self.lower_const(string)