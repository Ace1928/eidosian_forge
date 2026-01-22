import ctypes
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.singleton import S
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.decorator import doctest_depends_on
def _convert_expr(self, lj, expr):
    try:
        if len(expr) == 2:
            tmp_exprs = expr[0]
            final_exprs = expr[1]
            if len(final_exprs) != 1 and self.signature.ret_type == ctypes.c_double:
                raise NotImplementedError('Return of multiple expressions not supported for this callback')
            for name, e in tmp_exprs:
                val = lj._print(e)
                lj._add_tmp_var(name, val)
    except TypeError:
        final_exprs = [expr]
    vals = [lj._print(e) for e in final_exprs]
    return vals