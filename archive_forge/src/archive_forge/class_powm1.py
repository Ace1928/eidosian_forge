from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import cos, sin
class powm1(Function):
    """ Minus one plus x to the power of y, i.e. x**y - 1. For use when x is close to one or y is close to zero.

    Helper class for use with e.g. scipy.special.powm1
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.powm1.html
    """
    nargs = 2

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return Pow(self.args[0], self.args[1]) * self.args[1] / self.args[0]
        elif argindex == 2:
            return log(self.args[0]) * Pow(*self.args)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Pow(self, x, y, **kwargs):
        return _powm1(x, y)

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(Pow).evalf(*args, **kwargs)

    def _eval_simplify(self, **kwargs):
        x, y = self.args
        candidate = _powm1(x.simplify(**kwargs), y.simplify(**kwargs))
        if candidate != _powm1(x, y, evaluate=False):
            return candidate
        else:
            return powm1(x, y)