from .fixed_points import r13_fixed_points_of_psl2c_matrix # type: ignore
from ..hyperboloid import r13_dot, o13_inverse # type: ignore
from ..upper_halfspace import psl2c_to_o13 # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..sage_helper import _within_sage # type: ignore
def _safe_sqrt(p):
    """
    Compute the sqrt of a number that is known to be non-negative
    though might not be non-negative because of floating point
    issues. When using interval arithmetic, this means that
    while the upper bound will be non-negative, the lower bound
    we computed might be negative because it is too conservative.

    Example of a quantity that can be given to this function:
    negative inner product of two vectors in the positive
    light cone. This is because we know that the inner product
    of two such vectors is always non-positive.
    """
    if is_RealIntervalFieldElement(p):
        RIF = p.parent()
        p = p.intersection(RIF(0, sage.all.Infinity))
    elif p < 0:
        RF = p.parent()
        return RF(0)
    return p.sqrt()