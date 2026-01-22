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
def _apply_operator_Wavefunction(self, func, **options):
    from sympy.physics.quantum.state import Wavefunction
    var = self.variables
    wf_vars = func.args[1:]
    f = self.function
    new_expr = self.expr.subs(f, func(*var))
    new_expr = new_expr.doit()
    return Wavefunction(new_expr, *wf_vars)