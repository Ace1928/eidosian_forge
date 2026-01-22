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
def InRe(s, re):
    """Create regular expression membership test
    >>> re = Union(Re("a"),Re("b"))
    >>> print (simplify(InRe("a", re)))
    True
    >>> print (simplify(InRe("b", re)))
    True
    >>> print (simplify(InRe("c", re)))
    False
    """
    s = _coerce_seq(s, re.ctx)
    return BoolRef(Z3_mk_seq_in_re(s.ctx_ref(), s.as_ast(), re.as_ast()), s.ctx)