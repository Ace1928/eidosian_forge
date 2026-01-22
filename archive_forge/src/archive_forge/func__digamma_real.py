import operator
import math
import cmath
def _digamma_real(x):
    _intx = int(x)
    if _intx == x:
        if _intx <= 0:
            raise ZeroDivisionError('polygamma pole')
    if x < 0.5:
        x = 1.0 - x
        s = pi * cotpi(x)
    else:
        s = 0.0
    while x < 10.0:
        s -= 1.0 / x
        x += 1.0
    x2 = x ** (-2)
    t = x2
    for c in _psi_coeff:
        s -= c * t
        if t < 1e-20:
            break
        t *= x2
    return s + math_log(x) - 0.5 / x