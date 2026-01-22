import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _dpower(xc, xe, yc, ye, p):
    """Given integers xc, xe, yc and ye representing Decimals x = xc*10**xe and
    y = yc*10**ye, compute x**y.  Returns a pair of integers (c, e) such that:

      10**(p-1) <= c <= 10**p, and
      (c-1)*10**e < x**y < (c+1)*10**e

    in other words, c*10**e is an approximation to x**y with p digits
    of precision, and with an error in c of at most 1.  (This is
    almost, but not quite, the same as the error being < 1ulp: when c
    == 10**(p-1) we can only guarantee error < 10ulp.)

    We assume that: x is positive and not equal to 1, and y is nonzero.
    """
    b = len(str(abs(yc))) + ye
    lxc = _dlog(xc, xe, p + b + 1)
    shift = ye - b
    if shift >= 0:
        pc = lxc * yc * 10 ** shift
    else:
        pc = _div_nearest(lxc * yc, 10 ** (-shift))
    if pc == 0:
        if (len(str(xc)) + xe >= 1) == (yc > 0):
            coeff, exp = (10 ** (p - 1) + 1, 1 - p)
        else:
            coeff, exp = (10 ** p - 1, -p)
    else:
        coeff, exp = _dexp(pc, -(p + 1), p + 1)
        coeff = _div_nearest(coeff, 10)
        exp += 1
    return (coeff, exp)