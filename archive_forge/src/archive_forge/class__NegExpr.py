import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
class _NegExpr(Expr):

    def _str(self, *args, **kwargs):
        return '-%s' % args[0]._str(*args, **kwargs)

    def __repr__(self):
        return super(_NegExpr, self)._str(repr)

    def __call__(self, variables, backend=math, **kwargs):
        arg0, = self.all_args(variables, backend=backend, **kwargs)
        return -arg0

    def rate_coeff(self, *args, **kwargs):
        return (-self.args[0].rate_coeff(*args, **kwargs),)