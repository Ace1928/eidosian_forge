import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce
from numba.core import (
from numba.core.utils import (
from .rvsdg.bc2rvsdg import (
from .rvsdg.regionpasses import RegionVisitor
def _jump_if_not(self, pred):
    """Emit code for jump if predicate is false."""
    not_fn = ir.Const(operator.not_, loc=self.loc)
    res = ir.Expr.call(self.store(not_fn, '$not'), (self.vsmap[pred],), (), loc=self.loc)
    self.branch_predicate = self.store(res, '$jump_if')