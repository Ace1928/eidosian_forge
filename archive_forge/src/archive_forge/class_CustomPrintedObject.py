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
class CustomPrintedObject(Expr):

    def _lambdacode(self, printer):
        return 'lambda'

    def _tensorflowcode(self, printer):
        return 'tensorflow'

    def _numpycode(self, printer):
        return 'numpy'

    def _numexprcode(self, printer):
        return 'numexpr'

    def _mpmathcode(self, printer):
        return 'mpmath'