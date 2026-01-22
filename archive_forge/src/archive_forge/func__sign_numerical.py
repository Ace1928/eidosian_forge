import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
def _sign_numerical(self, prec):
    """
        Use interval arithmetics with precision prec to try to determine the
        sign. If we could not certify the sign, return None.
        The result is a pair (sign, interval).
        """
    RIF = RealIntervalField(prec)
    try:
        interval_val = RIF(self)
    except _SqrtException:
        return (None, None)
    if interval_val > 0:
        return (+1, interval_val)
    if interval_val < 0:
        return (-1, interval_val)
    return (None, interval_val)