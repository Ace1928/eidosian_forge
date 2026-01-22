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
def _inplace_binop(self, op, lhs, rhs, res):
    immuop = BINOPS_TO_OPERATORS[op]
    op = INPLACE_BINOPS_TO_OPERATORS[op + '=']
    lhs = self.get(lhs)
    rhs = self.get(rhs)
    expr = ir.Expr.inplace_binop(op, immuop, lhs=lhs, rhs=rhs, loc=self.loc)
    self.store(expr, res)