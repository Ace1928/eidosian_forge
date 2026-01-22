from sympy.core import Expr
from sympy.core.decorators import call_highest_priority, _sympifyit
from .fancysets import ImageSet
from .sets import set_add, set_sub, set_mul, set_div, set_pow, set_function
def _eval_func(self, func):
    res = set_function(func, self.set)
    if res is None:
        return SetExpr(ImageSet(func, self.set))
    return SetExpr(res)