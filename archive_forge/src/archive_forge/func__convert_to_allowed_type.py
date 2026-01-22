import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
def _convert_to_allowed_type(number):
    """
    When given a Python int, convert to Sage Integer (so that
    division of two integers gives a Rational). Otherwise,
    check that the type is allowed.
    """
    if isinstance(number, int):
        return Integer(number)
    if isinstance(number, Integer):
        return number
    if isinstance(number, Rational):
        return number
    if isinstance(number, NumberFieldElement):
        return number
    raise Exception('Not an allowed type')