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
class Simplifier:
    """Simplifiers act as pre-processing utilities for solvers.
    Build a custom simplifier and add it to a solver"""

    def __init__(self, simplifier, ctx=None):
        self.ctx = _get_ctx(ctx)
        self.simplifier = None
        if isinstance(simplifier, SimplifierObj):
            self.simplifier = simplifier
        elif isinstance(simplifier, list):
            simps = [Simplifier(s, ctx) for s in simplifier]
            self.simplifier = simps[0].simplifier
            for i in range(1, len(simps)):
                self.simplifier = Z3_simplifier_and_then(self.ctx.ref(), self.simplifier, simps[i].simplifier)
            Z3_simplifier_inc_ref(self.ctx.ref(), self.simplifier)
            return
        else:
            if z3_debug():
                _z3_assert(isinstance(simplifier, str), 'simplifier name expected')
            try:
                self.simplifier = Z3_mk_simplifier(self.ctx.ref(), str(simplifier))
            except Z3Exception:
                raise Z3Exception("unknown simplifier '%s'" % simplifier)
        Z3_simplifier_inc_ref(self.ctx.ref(), self.simplifier)

    def __deepcopy__(self, memo={}):
        return Simplifier(self.simplifier, self.ctx)

    def __del__(self):
        if self.simplifier is not None and self.ctx.ref() is not None and (Z3_simplifier_dec_ref is not None):
            Z3_simplifier_dec_ref(self.ctx.ref(), self.simplifier)

    def using_params(self, *args, **keys):
        """Return a simplifier that uses the given configuration options"""
        p = args2params(args, keys, self.ctx)
        return Simplifier(Z3_simplifier_using_params(self.ctx.ref(), self.simplifier, p.params), self.ctx)

    def add(self, solver):
        """Return a solver that applies the simplification pre-processing specified by the simplifier"""
        return Solver(Z3_solver_add_simplifier(self.ctx.ref(), solver.solver, self.simplifier), self.ctx)

    def help(self):
        """Display a string containing a description of the available options for the `self` simplifier."""
        print(Z3_simplifier_get_help(self.ctx.ref(), self.simplifier))

    def param_descrs(self):
        """Return the parameter description set."""
        return ParamDescrsRef(Z3_simplifier_get_param_descrs(self.ctx.ref(), self.simplifier), self.ctx)