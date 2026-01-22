import re
import warnings
from enum import Enum
from math import gcd
def as_eq(left, right):
    return Expr(Op.RELATIONAL, (RelOp.EQ, left, right))