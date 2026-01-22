from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def _normalize_points(a, b):
    """
    Reduce the number of cases involving infinity that we need to
    consider.

    In particular (assuming no degeneracy), a[0], a[1] and b[0] are
    never infinite.
    """
    a_infinities = [i for i, z in enumerate(a) if z == Infinity]
    if len(a_infinities) > 0:
        i = a_infinities[0]
        a, b = (a[i:] + a[:i], b[i:] + b[:i])
    b_infinities = [i for i, z in enumerate(b) if z == Infinity]
    if len(b_infinities) > 0:
        i = b_infinities[0]
        if a[0] != Infinity:
            a, b = (a[i:] + a[:i], b[i:] + b[:i])
        elif i == 2:
            a, b = ([a[0], a[2], a[1]], [b[0], b[2], b[1]])
    (a.reverse(), b.reverse())
    return (a, b)