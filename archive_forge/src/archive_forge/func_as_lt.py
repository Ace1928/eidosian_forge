import re
import warnings
from enum import Enum
from math import gcd
def as_lt(left, right):
    return Expr(Op.RELATIONAL, (RelOp.LT, left, right))