import re
import warnings
from enum import Enum
from math import gcd
def as_ref(expr):
    """Return object as referencing expression.
    """
    return Expr(Op.REF, expr)