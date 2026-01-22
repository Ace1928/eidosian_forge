from sympy.core.basic import Basic
from sympy.core.symbol import (Symbol, symbols)
from sympy.utilities.lambdify import lambdify
from .util import interpolate, rinterpolate, create_bounds, update_bounds
from sympy.utilities.iterables import sift
def _interpret_args(self, args):
    f, gradient = (None, self.gradient)
    atoms, lists = self._sort_args(args)
    s = self._pop_symbol_list(lists)
    s = self._fill_in_vars(s)
    f_str = ', '.join((str(fa) for fa in atoms))
    s_str = (str(sa) for sa in s)
    s_str = ', '.join((sa for sa in s_str if sa.find('unbound') < 0))
    f_error = ValueError('Could not interpret arguments %s as functions of %s.' % (f_str, s_str))
    if len(atoms) == 1:
        fv = atoms[0]
        try:
            f = lambdify(s, [fv, fv, fv])
        except TypeError:
            raise f_error
    elif len(atoms) == 3:
        fr, fg, fb = atoms
        try:
            f = lambdify(s, [fr, fg, fb])
        except TypeError:
            raise f_error
    else:
        raise ValueError('A ColorScheme must provide 1 or 3 functions in x, y, z, u, and/or v.')
    if len(lists) == 0:
        gargs = []
    elif len(lists) == 1:
        gargs = lists[0]
    elif len(lists) == 2:
        try:
            (r1, g1, b1), (r2, g2, b2) = lists
        except TypeError:
            raise ValueError('If two color arguments are given, they must be given in the format (r1, g1, b1), (r2, g2, b2).')
        gargs = lists
    elif len(lists) == 3:
        try:
            (r1, r2), (g1, g2), (b1, b2) = lists
        except Exception:
            raise ValueError('If three color arguments are given, they must be given in the format (r1, r2), (g1, g2), (b1, b2). To create a multi-step gradient, use the syntax [0, colorStart, step1, color1, ..., 1, colorEnd].')
        gargs = [[r1, g1, b1], [r2, g2, b2]]
    else:
        raise ValueError("Don't know what to do with collection arguments %s." % ', '.join((str(l) for l in lists)))
    if gargs:
        try:
            gradient = ColorGradient(*gargs)
        except Exception as ex:
            raise ValueError('Could not initialize a gradient with arguments %s. Inner exception: %s' % (gargs, str(ex)))
    return (f, gradient)