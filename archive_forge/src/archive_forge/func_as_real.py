import re
import warnings
from enum import Enum
from math import gcd
def as_real(obj, kind=4):
    """Return object as REAL constant.
    """
    if isinstance(obj, int):
        return Expr(Op.REAL, (float(obj), kind))
    if isinstance(obj, float):
        return Expr(Op.REAL, (obj, kind))
    if isinstance(obj, Expr):
        if obj.op is Op.REAL:
            return obj
        elif obj.op is Op.INTEGER:
            return Expr(Op.REAL, (float(obj.data[0]), kind))
    raise OpError(f'cannot convert {obj} to REAL constant')