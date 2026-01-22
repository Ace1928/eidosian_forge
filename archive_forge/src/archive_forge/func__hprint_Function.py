from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING
import itertools
from sympy.core import Add, Float, Mod, Mul, Number, S, Symbol, Expr
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import Function, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true, BooleanTrue, BooleanFalse
from sympy.tensor.array import NDimArray
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE
from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str
from sympy.utilities.iterables import has_variety, sift
import re
def _hprint_Function(self, func: str) -> str:
    """
        Logic to decide how to render a function to latex
          - if it is a recognized latex name, use the appropriate latex command
          - if it is a single letter, excluding sub- and superscripts, just use that letter
          - if it is a longer name, then put \\operatorname{} around it and be
            mindful of undercores in the name
        """
    func = self._deal_with_super_sub(func)
    superscriptidx = func.find('^')
    subscriptidx = func.find('_')
    if func in accepted_latex_functions:
        name = '\\%s' % func
    elif len(func) == 1 or func.startswith('\\') or subscriptidx == 1 or (superscriptidx == 1):
        name = func
    elif superscriptidx > 0 and subscriptidx > 0:
        name = '\\operatorname{%s}%s' % (func[:min(subscriptidx, superscriptidx)], func[min(subscriptidx, superscriptidx):])
    elif superscriptidx > 0:
        name = '\\operatorname{%s}%s' % (func[:superscriptidx], func[superscriptidx:])
    elif subscriptidx > 0:
        name = '\\operatorname{%s}%s' % (func[:subscriptidx], func[subscriptidx:])
    else:
        name = '\\operatorname{%s}' % func
    return name