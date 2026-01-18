import random
from mpmath import *
from mpmath.libmp import *
def test_str_prec0():
    assert to_str(from_float(1.234), 0) == '.0e+0'
    assert to_str(from_float(1e-15), 0) == '.0e-15'
    assert to_str(from_float(1000000000000000.0), 0) == '.0e+15'
    assert to_str(from_float(-1e-15), 0) == '-.0e-15'
    assert to_str(from_float(-1000000000000000.0), 0) == '-.0e+15'