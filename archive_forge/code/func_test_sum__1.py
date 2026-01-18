from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises
from sympy.printing.tensorflow import TensorflowPrinter
from sympy.printing.lambdarepr import lambdarepr, LambdaPrinter, NumExprPrinter
def test_sum__1():
    s = Sum(x ** i, (i, a, b))
    l = lambdarepr(s)
    assert l == '(builtins.sum(x**i for i in range(a, b+1)))'
    args = (x, a, b)
    f = lambdify(args, s)
    v = (2, 3, 8)
    assert f(*v) == s.subs(zip(args, v)).doit()