import re
import warnings
from enum import Enum
from math import gcd
def as_deref(expr):
    """Return object as dereferencing expression.
    """
    return Expr(Op.DEREF, expr)