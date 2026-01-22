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
def RecAddDefinition(f, args, body):
    """Set the body of a recursive function.
       Recursive definitions can be simplified if they are applied to ground
       arguments.
    >>> ctx = Context()
    >>> fac = RecFunction('fac', IntSort(ctx), IntSort(ctx))
    >>> n = Int('n', ctx)
    >>> RecAddDefinition(fac, n, If(n == 0, 1, n*fac(n-1)))
    >>> simplify(fac(5))
    120
    >>> s = Solver(ctx=ctx)
    >>> s.add(fac(n) < 3)
    >>> s.check()
    sat
    >>> s.model().eval(fac(5))
    120
    """
    if is_app(args):
        args = [args]
    ctx = body.ctx
    args = _get_args(args)
    n = len(args)
    _args = (Ast * n)()
    for i in range(n):
        _args[i] = args[i].ast
    Z3_add_rec_def(ctx.ref(), f.ast, n, _args, body.ast)