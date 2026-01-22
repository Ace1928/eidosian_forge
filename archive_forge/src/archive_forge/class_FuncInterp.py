from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class FuncInterp(Z3PPObject):
    """Stores the interpretation of a function in a Z3 model."""

    def __init__(self, f, ctx):
        self.f = f
        self.ctx = ctx
        if self.f is not None:
            Z3_func_interp_inc_ref(self.ctx.ref(), self.f)

    def __del__(self):
        if self.f is not None and self.ctx.ref() is not None and (Z3_func_interp_dec_ref is not None):
            Z3_func_interp_dec_ref(self.ctx.ref(), self.f)

    def else_value(self):
        """
        Return the `else` value for a function interpretation.
        Return None if Z3 did not specify the `else` value for
        this object.

        >>> f = Function('f', IntSort(), IntSort())
        >>> s = Solver()
        >>> s.add(f(0) == 1, f(1) == 1, f(2) == 0)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m[f]
        [2 -> 0, else -> 1]
        >>> m[f].else_value()
        1
        """
        r = Z3_func_interp_get_else(self.ctx.ref(), self.f)
        if r:
            return _to_expr_ref(r, self.ctx)
        else:
            return None

    def num_entries(self):
        """Return the number of entries/points in the function interpretation `self`.

        >>> f = Function('f', IntSort(), IntSort())
        >>> s = Solver()
        >>> s.add(f(0) == 1, f(1) == 1, f(2) == 0)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m[f]
        [2 -> 0, else -> 1]
        >>> m[f].num_entries()
        1
        """
        return int(Z3_func_interp_get_num_entries(self.ctx.ref(), self.f))

    def arity(self):
        """Return the number of arguments for each entry in the function interpretation `self`.

        >>> f = Function('f', IntSort(), IntSort())
        >>> s = Solver()
        >>> s.add(f(0) == 1, f(1) == 1, f(2) == 0)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m[f].arity()
        1
        """
        return int(Z3_func_interp_get_arity(self.ctx.ref(), self.f))

    def entry(self, idx):
        """Return an entry at position `idx < self.num_entries()` in the function interpretation `self`.

        >>> f = Function('f', IntSort(), IntSort())
        >>> s = Solver()
        >>> s.add(f(0) == 1, f(1) == 1, f(2) == 0)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m[f]
        [2 -> 0, else -> 1]
        >>> m[f].num_entries()
        1
        >>> m[f].entry(0)
        [2, 0]
        """
        if idx >= self.num_entries():
            raise IndexError
        return FuncEntry(Z3_func_interp_get_entry(self.ctx.ref(), self.f, idx), self.ctx)

    def translate(self, other_ctx):
        """Copy model 'self' to context 'other_ctx'.
        """
        return ModelRef(Z3_model_translate(self.ctx.ref(), self.model, other_ctx.ref()), other_ctx)

    def __copy__(self):
        return self.translate(self.ctx)

    def __deepcopy__(self, memo={}):
        return self.translate(self.ctx)

    def as_list(self):
        """Return the function interpretation as a Python list.
        >>> f = Function('f', IntSort(), IntSort())
        >>> s = Solver()
        >>> s.add(f(0) == 1, f(1) == 1, f(2) == 0)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m[f]
        [2 -> 0, else -> 1]
        >>> m[f].as_list()
        [[2, 0], 1]
        """
        r = [self.entry(i).as_list() for i in range(self.num_entries())]
        r.append(self.else_value())
        return r

    def __repr__(self):
        return obj_to_string(self)