from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.cache import cacheit
def _ith_point(self, i):
    """
        Returns the i'th point of a series
        If start point is negative infinity, point is returned from the end.
        Assumes the first point to be indexed zero.

        Examples
        ========

        TODO
        """
    if self.start is S.NegativeInfinity:
        initial = self.stop
        step = -1
    else:
        initial = self.start
        step = 1
    return initial + i * step