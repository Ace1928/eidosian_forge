from sympy.core.basic import Basic
from sympy.core.symbol import (Symbol, symbols)
from sympy.utilities.lambdify import lambdify
from .util import interpolate, rinterpolate, create_bounds, update_bounds
from sympy.utilities.iterables import sift
def _fill_in_vars(self, args):
    defaults = symbols('x,y,z,u,v')
    v_error = ValueError('Could not find what to plot.')
    if len(args) == 0:
        return defaults
    if not isinstance(args, (tuple, list)):
        raise v_error
    if len(args) == 0:
        return defaults
    for s in args:
        if s is not None and (not isinstance(s, Symbol)):
            raise v_error
    vars = [Symbol('unbound%i' % i) for i in range(1, 6)]
    if len(args) == 1:
        vars[3] = args[0]
    elif len(args) == 2:
        if args[0] is not None:
            vars[3] = args[0]
        if args[1] is not None:
            vars[4] = args[1]
    elif len(args) >= 3:
        if args[0] is not None:
            vars[0] = args[0]
        if args[1] is not None:
            vars[1] = args[1]
        if args[2] is not None:
            vars[2] = args[2]
        if len(args) >= 4:
            vars[3] = args[3]
            if len(args) >= 5:
                vars[4] = args[4]
    return vars