import string
from ..sage_helper import _within_sage, sage_method
def univ_abs(z):
    """
    Compute a reasonable choice for the absolute value of z.
    """
    try:
        return z.abs()
    except (TypeError, AttributeError):
        if hasattr(z, 'coefficients'):
            return max([0] + [univ_abs(c) for c in z.coefficients()])
        else:
            return 0 if z == 0 else Infinity