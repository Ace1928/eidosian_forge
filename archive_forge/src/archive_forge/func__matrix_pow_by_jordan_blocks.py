import mpmath as mp
from collections.abc import Callable
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import diff
from sympy.core.expr import Expr
from sympy.core.kind import _NumberKind, UndefinedKind
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, uniquely_named_symbol
from sympy.core.sympify import sympify, _sympify
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta, LeviCivita
from sympy.polys import cancel
from sympy.printing import sstr
from sympy.printing.defaults import Printable
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import flatten, NotIterable, is_sequence, reshape
from sympy.utilities.misc import as_int, filldedent
from .common import (
from .utilities import _iszero, _is_zero_after_expand_mul, _simplify
from .determinant import (
from .reductions import _is_echelon, _echelon_form, _rank, _rref
from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize
from .eigen import (
from .decompositions import (
from .graph import (
from .solvers import (
from .inverse import (
def _matrix_pow_by_jordan_blocks(self, num):
    from sympy.matrices import diag, MutableMatrix

    def jordan_cell_power(jc, n):
        N = jc.shape[0]
        l = jc[0, 0]
        if l.is_zero:
            if N == 1 and n.is_nonnegative:
                jc[0, 0] = l ** n
            elif not (n.is_integer and n.is_nonnegative):
                raise NonInvertibleMatrixError('Non-invertible matrix can only be raised to a nonnegative integer')
            else:
                for i in range(N):
                    jc[0, i] = KroneckerDelta(i, n)
        else:
            for i in range(N):
                bn = binomial(n, i)
                if isinstance(bn, binomial):
                    bn = bn._eval_expand_func()
                jc[0, i] = l ** (n - i) * bn
        for i in range(N):
            for j in range(1, N - i):
                jc[j, i + j] = jc[j - 1, i + j - 1]
    P, J = self.jordan_form()
    jordan_cells = J.get_diag_blocks()
    jordan_cells = [MutableMatrix(j) for j in jordan_cells]
    for j in jordan_cells:
        jordan_cell_power(j, num)
    return self._new(P.multiply(diag(*jordan_cells)).multiply(P.inv()))