from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_ArrayContraction(self, expr):
    from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
    base = expr.expr
    contraction_indices = expr.contraction_indices
    if isinstance(base, ArrayTensorProduct):
        elems = ','.join(['%s' % self._print(arg) for arg in base.args])
        ranks = base.subranks
    else:
        elems = self._print(base)
        ranks = [len(base.shape)]
    contraction_string, letters_free, letters_dum = self._get_einsum_string(ranks, contraction_indices)
    if not contraction_indices:
        return self._print(base)
    if isinstance(base, ArrayTensorProduct):
        elems = ','.join(['%s' % self._print(arg) for arg in base.args])
    else:
        elems = self._print(base)
    return '%s("%s", %s)' % (self._module_format(self._module + '.' + self._einsum), '{}->{}'.format(contraction_string, ''.join(sorted(letters_free))), elems)