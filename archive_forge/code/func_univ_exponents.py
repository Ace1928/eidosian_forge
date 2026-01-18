import string
from ..sage_helper import _within_sage, sage_method
def univ_exponents(p):
    try:
        return [a[0] for a in p.exponents()]
    except TypeError:
        return p.exponents()