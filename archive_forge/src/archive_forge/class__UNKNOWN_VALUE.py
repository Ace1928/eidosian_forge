import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
class _UNKNOWN_VALUE(object):
    """Represents an unknown value, this is for ease of debugging purposes only.
    """

    def __init__(self, varname):
        self._varname = varname

    def __repr__(self):
        return '_UNKNOWN_VALUE({})'.format(self._varname)