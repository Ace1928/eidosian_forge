import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
def all_args(self, variables, backend=math, evaluate=True, **kwargs):
    if self.nargs is None or self.nargs == -1:
        nargs = len(self.args)
    else:
        nargs = self.nargs
    return [self.arg(variables, i, backend, evaluate, **kwargs) for i in range(nargs)]