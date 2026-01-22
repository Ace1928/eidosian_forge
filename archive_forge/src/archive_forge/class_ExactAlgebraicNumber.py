from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
class ExactAlgebraicNumber(ApproximateAlgebraicNumber):
    """
    An ApproximateAlgebraicNumber which is specified
    explicitly by its minimal polynomial.
    """

    def __init__(self, poly, approx_root):
        if not acceptable_error(poly, approx_root, ZZ(0), 0.2):
            raise ValueError('Given number does not seem to be a root of this polynomial')
        self._min_poly = poly
        self._approx_root = approx_root

    @cached_method
    def __call__(self, prec):
        roots = [r[0] for r in self._min_poly.roots(ComplexField(prec))]

        def dist_to_defining_root(z):
            return abs(z - self._approx_root)
        return sorted(roots, key=dist_to_defining_root)[0]