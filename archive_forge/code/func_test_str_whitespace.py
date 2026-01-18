import random
from mpmath import *
from mpmath.libmp import *
def test_str_whitespace():
    assert mpf('1.26 ') == 1.26