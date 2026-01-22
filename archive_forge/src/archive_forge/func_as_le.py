import re
import warnings
from enum import Enum
from math import gcd
def as_le(left, right):
    return Expr(Op.RELATIONAL, (RelOp.LE, left, right))