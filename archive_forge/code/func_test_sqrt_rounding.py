from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_sqrt_rounding():
    for i in [2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]:
        i = from_int(i)
        for dps in [7, 15, 83, 106, 2000]:
            mp.dps = dps
            a = mpf_pow_int(mpf_sqrt(i, mp.prec, round_down), 2, mp.prec, round_down)
            b = mpf_pow_int(mpf_sqrt(i, mp.prec, round_up), 2, mp.prec, round_up)
            assert mpf_lt(a, i)
            assert mpf_gt(b, i)
    random.seed(1234)
    prec = 100
    for rnd in [round_down, round_nearest, round_ceiling]:
        for i in range(100):
            a = mpf_rand(prec)
            b = mpf_mul(a, a)
            assert mpf_sqrt(b, prec, rnd) == a
    mp.dps = 100
    a = mpf(9) + 1e-90
    b = mpf(9) - 1e-90
    mp.dps = 15
    assert sqrt(a, rounding='d') == 3
    assert sqrt(a, rounding='n') == 3
    assert sqrt(a, rounding='u') > 3
    assert sqrt(b, rounding='d') < 3
    assert sqrt(b, rounding='n') == 3
    assert sqrt(b, rounding='u') == 3
    assert sqrt(mpf('7.0503726185518891')) == mpf('2.655253776675949')