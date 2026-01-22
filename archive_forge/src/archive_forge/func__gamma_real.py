import operator
import math
import cmath
def _gamma_real(x):
    _intx = int(x)
    if _intx == x:
        if _intx <= 0:
            raise ZeroDivisionError('gamma function pole')
        if _intx <= _max_exact_gamma:
            return _exact_gamma[_intx]
    if x < 0.5:
        return pi / (_sinpi_real(x) * _gamma_real(1 - x))
    else:
        x -= 1.0
        r = _lanczos_p[0]
        for i in range(1, _lanczos_g + 2):
            r += _lanczos_p[i] / (x + i)
        t = x + _lanczos_g + 0.5
        return 2.5066282746310007 * t ** (x + 0.5) * math.exp(-t) * r