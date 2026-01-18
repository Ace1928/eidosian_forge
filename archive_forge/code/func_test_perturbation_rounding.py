from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_perturbation_rounding():
    mp.dps = 100
    a = pi / 10 ** 50
    b = -pi / 10 ** 50
    c = 1 + a
    d = 1 + b
    mp.dps = 15
    assert exp(a) == 1
    assert exp(a, rounding='c') > 1
    assert exp(b, rounding='c') == 1
    assert exp(a, rounding='f') == 1
    assert exp(b, rounding='f') < 1
    assert cos(a) == 1
    assert cos(a, rounding='c') == 1
    assert cos(b, rounding='c') == 1
    assert cos(a, rounding='f') < 1
    assert cos(b, rounding='f') < 1
    for f in [sin, atan, asinh, tanh]:
        assert f(a) == +a
        assert f(a, rounding='c') > a
        assert f(a, rounding='f') < a
        assert f(b) == +b
        assert f(b, rounding='c') > b
        assert f(b, rounding='f') < b
    for f in [asin, tan, sinh, atanh]:
        assert f(a) == +a
        assert f(b) == +b
        assert f(a, rounding='c') > a
        assert f(b, rounding='c') > b
        assert f(a, rounding='f') < a
        assert f(b, rounding='f') < b
    assert ln(c) == +a
    assert ln(d) == +b
    assert ln(c, rounding='c') > a
    assert ln(c, rounding='f') < a
    assert ln(d, rounding='c') > b
    assert ln(d, rounding='f') < b
    assert cosh(a) == 1
    assert cosh(b) == 1
    assert cosh(a, rounding='c') > 1
    assert cosh(b, rounding='c') > 1
    assert cosh(a, rounding='f') == 1
    assert cosh(b, rounding='f') == 1