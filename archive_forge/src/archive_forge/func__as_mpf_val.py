from sympy.core.numbers import NumberSymbol
from sympy.core.singleton import Singleton
from sympy.printing.pretty.stringpict import prettyForm
import mpmath.libmp as mlib
def _as_mpf_val(self, prec):
    return mlib.from_float(1.05457162e-34, prec)