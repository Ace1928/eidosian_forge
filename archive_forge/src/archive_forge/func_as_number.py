import re
import warnings
from enum import Enum
from math import gcd
def as_number(obj, kind=4):
    """Return object as INTEGER or REAL constant.
    """
    if isinstance(obj, int):
        return Expr(Op.INTEGER, (obj, kind))
    if isinstance(obj, float):
        return Expr(Op.REAL, (obj, kind))
    if isinstance(obj, Expr):
        if obj.op in (Op.INTEGER, Op.REAL):
            return obj
    raise OpError(f'cannot convert {obj} to INTEGER or REAL constant')