from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def BoolVector(prefix, sz, ctx=None):
    """Return a list of Boolean constants of size `sz`.

    The constants are named using the given prefix.
    If `ctx=None`, then the global context is used.

    >>> P = BoolVector('p', 3)
    >>> P
    [p__0, p__1, p__2]
    >>> And(P)
    And(p__0, p__1, p__2)
    """
    return [Bool('%s__%s' % (prefix, i)) for i in range(sz)]