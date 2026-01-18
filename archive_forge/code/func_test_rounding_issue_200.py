from mpmath import *
from mpmath.libmp import *
def test_rounding_issue_200():
    a = from_man_exp(9867, -100)
    b = from_man_exp(9867, -200)
    c = from_man_exp(-1, 0)
    z = (1, 1023, -10, 10)
    assert mpf_add(a, c, 10, 'd') == z
    assert mpf_add(b, c, 10, 'd') == z
    assert mpf_add(c, a, 10, 'd') == z
    assert mpf_add(c, b, 10, 'd') == z