import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _dexp(c, e, p):
    """Compute an approximation to exp(c*10**e), with p decimal places of
    precision.

    Returns integers d, f such that:

      10**(p-1) <= d <= 10**p, and
      (d-1)*10**f < exp(c*10**e) < (d+1)*10**f

    In other words, d*10**f is an approximation to exp(c*10**e) with p
    digits of precision, and with an error in d of at most 1.  This is
    almost, but not quite, the same as the error being < 1ulp: when d
    = 10**(p-1) the error could be up to 10 ulp."""
    p += 2
    extra = max(0, e + len(str(c)) - 1)
    q = p + extra
    shift = e + q
    if shift >= 0:
        cshift = c * 10 ** shift
    else:
        cshift = c // 10 ** (-shift)
    quot, rem = divmod(cshift, _log10_digits(q))
    rem = _div_nearest(rem, 10 ** extra)
    return (_div_nearest(_iexp(rem, 10 ** p), 1000), quot - p + 3)