from sympy.core import S
from sympy.core.function import Lambda
from sympy.core.power import Pow
from .pycode import PythonCodePrinter, _known_functions_math, _print_known_const, _print_known_func, _unpack_integral_limits, ArrayPrinter
from .codeprinter import CodePrinter
def _print_arg(self, expr):
    return '%s(%s)' % (self._module_format(self._module + '.angle'), self._print(expr.args[0]))