import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _ilog(x, M, L=8):
    """Integer approximation to M*log(x/M), with absolute error boundable
    in terms only of x/M.

    Given positive integers x and M, return an integer approximation to
    M * log(x/M).  For L = 8 and 0.1 <= x/M <= 10 the difference
    between the approximation and the exact result is at most 22.  For
    L = 8 and 1.0 <= x/M <= 10.0 the difference is at most 15.  In
    both cases these are upper bounds on the error; it will usually be
    much smaller."""
    y = x - M
    R = 0
    while R <= L and abs(y) << L - R >= M or (R > L and abs(y) >> R - L >= M):
        y = _div_nearest(M * y << 1, M + _sqrt_nearest(M * (M + _rshift_nearest(y, R)), M))
        R += 1
    T = -int(-10 * len(str(M)) // (3 * L))
    yshift = _rshift_nearest(y, R)
    w = _div_nearest(M, T)
    for k in range(T - 1, 0, -1):
        w = _div_nearest(M, k) - _div_nearest(yshift * w, M)
    return _div_nearest(w * y, M)