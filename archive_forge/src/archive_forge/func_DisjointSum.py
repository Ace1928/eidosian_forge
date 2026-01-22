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
def DisjointSum(name, sorts, ctx=None):
    """Create a named tagged union sort base on a set of underlying sorts
    Example:
        >>> sum, ((inject0, extract0), (inject1, extract1)) = DisjointSum("+", [IntSort(), StringSort()])
    """
    sum = Datatype(name, ctx)
    for i in range(len(sorts)):
        sum.declare('inject%d' % i, ('project%d' % i, sorts[i]))
    sum = sum.create()
    return (sum, [(sum.constructor(i), sum.accessor(i, 0)) for i in range(len(sorts))])