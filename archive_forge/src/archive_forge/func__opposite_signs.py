import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
def _opposite_signs(left, right, prec):
    """
    Given two objects left and right that can be coerced to real interval of
    the given precision, try to certify their signs. If succeed, return True
    if the signs are opposite and False otherwise. If failed, return None.
    """
    RIF = RealIntervalField(prec)
    try:
        left_interval = RIF(left)
        right_interval = RIF(right)
    except _SqrtException:
        return None
    left_negative = bool(left_interval < 0)
    left_positive = bool(left_interval > 0)
    left_determined = left_negative or left_positive
    right_negative = bool(right_interval < 0)
    right_positive = bool(right_interval > 0)
    right_determined = right_negative or right_positive
    if left_determined and right_determined:
        return left_positive ^ right_positive
    return None