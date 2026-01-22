import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _html_op_name(a):
    global _z3_html_op_to_str
    if isinstance(a, z3.FuncDeclRef):
        f = a
    else:
        f = a.decl()
    k = f.kind()
    n = _z3_html_op_to_str.get(k, None)
    if n is None:
        sym = Z3_get_decl_name(f.ctx_ref(), f.ast)
        if Z3_get_symbol_kind(f.ctx_ref(), sym) == Z3_INT_SYMBOL:
            return '&#950;<sub>%s</sub>' % Z3_get_symbol_int(f.ctx_ref(), sym)
        else:
            return f.name()
    else:
        return n