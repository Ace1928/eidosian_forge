from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, expand)
from sympy.core.mul import Mul
from sympy.core.numbers import oo
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
from sympy.matrices import eye
def _print_pretty(self, printer, *args):
    pform = self._print_operator_name_pretty(printer, *args)
    label_pform = self._print_label_pretty(printer, *args)
    label_pform = prettyForm(*label_pform.parens(left='(', right=')'))
    pform = prettyForm(*pform.right(label_pform))
    return pform