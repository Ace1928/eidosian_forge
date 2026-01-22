import math
import logging
from pyomo.common.errors import InfeasibleConstraintException, IntervalException
def _inverse_power2(zl, zu, xl, xu, feasiblity_tol):
    """z = x**y => compute bounds on y
    y = ln(z) / ln(x)

    This function assumes the exponent can be fractional, so x must be
    positive. This method should not be called if the exponent is an
    integer.

    """
    if xu <= 0:
        raise IntervalException('Cannot raise a negative variable to a fractional power.')
    if xl > 0 and zu <= 0 or (xl >= 0 and zu < 0):
        raise InfeasibleConstraintException('A positive variable raised to the power of anything must be positive.')
    lba, uba = log(zl, zu)
    lbb, ubb = log(xl, xu)
    yl, yu = div(lba, uba, lbb, ubb, feasiblity_tol)
    return (yl, yu)