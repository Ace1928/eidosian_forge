from .pycode import (
from .numpy import NumPyPrinter  # NumPyPrinter is imported for backward compatibility
from sympy.core.sorting import default_sort_key
def _print_NumExprEvaluate(self, expr):
    evaluate = self._module_format(self.module + '.evaluate')
    return "%s('%s', truediv=True)" % (evaluate, self._print(expr.expr))