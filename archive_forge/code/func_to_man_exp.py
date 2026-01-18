import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def to_man_exp(s):
    """Return (man, exp) of a raw mpf. Raise an error if inf/nan."""
    sign, man, exp, bc = s
    if not man and exp:
        raise ValueError('mantissa and exponent are undefined for %s' % man)
    return (man, exp)