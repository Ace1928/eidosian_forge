import math
import functools
import logging
Round to nearest multiple of factor and return shortest decimal representation.

    This chooses the float that is closer to a multiple of the given factor while
    having the shortest decimal representation (the least number of fractional decimal
    digits).

    For example, given the following:

    >>> nearestMultipleShortestRepr(-0.61883544921875, 1.0/(1<<14))
    '-0.61884'

    Useful when you need to serialize or print a fixed-point number (or multiples
    thereof, such as F2Dot14 fractions of 180 degrees in COLRv1 PaintRotate) in
    a human-readable form.

    Args:
        value (value): The value to be rounded and serialized.
        factor (float): The value which the result is a close multiple of.

    Returns:
        str: A compact string representation of the value.
    