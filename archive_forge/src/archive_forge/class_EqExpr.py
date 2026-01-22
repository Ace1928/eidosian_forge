import math
from ..util.pyutil import deprecated
from ..util._expr import Expr
@deprecated(use_instead=MassActionEq)
class EqExpr(Expr):
    """Baseclass for equilibrium expressions"""
    kw = {'eq': None, 'ref': None}