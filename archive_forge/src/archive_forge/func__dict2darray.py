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
def _dict2darray(decls, ctx):
    sz = len(decls)
    _names = (Symbol * sz)()
    _decls = (FuncDecl * sz)()
    i = 0
    for k in decls:
        v = decls[k]
        if z3_debug():
            _z3_assert(isinstance(k, str), 'String expected')
            _z3_assert(is_func_decl(v) or is_const(v), 'Z3 declaration or constant expected')
        _names[i] = to_symbol(k, ctx)
        if is_const(v):
            _decls[i] = v.decl().ast
        else:
            _decls[i] = v.ast
        i = i + 1
    return (sz, _names, _decls)