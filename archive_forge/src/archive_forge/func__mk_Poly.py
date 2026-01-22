from functools import reduce
import math
from operator import add
from ._expr import Expr
def _mk_Poly(parameter_name, reciprocal=False, shift_name='shift'):
    """Class factory of Expr subclass for (shifted) polynomial

    Parameters
    ----------
    parameter: str
        name of parameter
    reciprocal: bool
        whether the polynomial is in the reciprocal of the parameter

    Returns
    -------
    Expr subclass for a shifted polynomial with the args: offset, p0, p1, ...
    the class has the method "eval_poly" with same signature as __call__


    Examples
    --------
    >>> P = _mk_Poly('x')
    >>> p = P([3, 5, 7, 2])
    >>> p.eval_poly({'x': 13}) == 5 + 7*(13-3) + 2*(13-3)**2
    True

    """

    class Poly(Expr):
        """Args: shift, p0, p1, ..."""
        argument_names = (shift_name, Ellipsis)
        parameter_keys = (parameter_name,)
        skip_poly = 0

        def eval_poly(self, variables, backend=math):
            all_args = self.all_args(variables, backend=backend)
            x = variables[parameter_name]
            offset, coeffs = (all_args[self.skip_poly], all_args[self.skip_poly + 1:])
            return _eval_poly(x, offset, coeffs, reciprocal)
    return Poly