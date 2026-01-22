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
def MultiPattern(*args):
    """Create a Z3 multi-pattern using the given expressions `*args`

    >>> f = Function('f', IntSort(), IntSort())
    >>> g = Function('g', IntSort(), IntSort())
    >>> x = Int('x')
    >>> q = ForAll(x, f(x) != g(x), patterns = [ MultiPattern(f(x), g(x)) ])
    >>> q
    ForAll(x, f(x) != g(x))
    >>> q.num_patterns()
    1
    >>> is_pattern(q.pattern(0))
    True
    >>> q.pattern(0)
    MultiPattern(f(Var(0)), g(Var(0)))
    """
    if z3_debug():
        _z3_assert(len(args) > 0, 'At least one argument expected')
        _z3_assert(all([is_expr(a) for a in args]), 'Z3 expressions expected')
    ctx = args[0].ctx
    args, sz = _to_ast_array(args)
    return PatternRef(Z3_mk_pattern(ctx.ref(), sz, args), ctx)