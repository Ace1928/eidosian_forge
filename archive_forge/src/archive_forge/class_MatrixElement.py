from __future__ import annotations
from functools import wraps
from sympy.core import S, Integer, Basic, Mul, Add
from sympy.core.assumptions import check_assumptions
from sympy.core.decorators import call_highest_priority
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.logic import FuzzyBool
from sympy.core.symbol import Str, Dummy, symbols, Symbol
from sympy.core.sympify import SympifyError, _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.matrices import MatrixKind, MatrixBase
from sympy.multipledispatch import dispatch
from sympy.utilities.misc import filldedent
from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
from .special import ZeroMatrix, Identity
from .determinant import Determinant
class MatrixElement(Expr):
    parent = property(lambda self: self.args[0])
    i = property(lambda self: self.args[1])
    j = property(lambda self: self.args[2])
    _diff_wrt = True
    is_symbol = True
    is_commutative = True

    def __new__(cls, name, n, m):
        n, m = map(_sympify, (n, m))
        from sympy.matrices.matrices import MatrixBase
        if isinstance(name, str):
            name = Symbol(name)
        else:
            if isinstance(name, MatrixBase):
                if n.is_Integer and m.is_Integer:
                    return name[n, m]
                name = _sympify(name)
            else:
                name = _sympify(name)
                if not isinstance(name.kind, MatrixKind):
                    raise TypeError('First argument of MatrixElement should be a matrix')
            if not getattr(name, 'valid_index', lambda n, m: True)(n, m):
                raise IndexError('indices out of range')
        obj = Expr.__new__(cls, name, n, m)
        return obj

    @property
    def symbol(self):
        return self.args[0]

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args
        return args[0][args[1], args[2]]

    @property
    def indices(self):
        return self.args[1:]

    def _eval_derivative(self, v):
        if not isinstance(v, MatrixElement):
            from sympy.matrices.matrices import MatrixBase
            if isinstance(self.parent, MatrixBase):
                return self.parent.diff(v)[self.i, self.j]
            return S.Zero
        M = self.args[0]
        m, n = self.parent.shape
        if M == v.args[0]:
            return KroneckerDelta(self.args[1], v.args[1], (0, m - 1)) * KroneckerDelta(self.args[2], v.args[2], (0, n - 1))
        if isinstance(M, Inverse):
            from sympy.concrete.summations import Sum
            i, j = self.args[1:]
            i1, i2 = symbols('z1, z2', cls=Dummy)
            Y = M.args[0]
            r1, r2 = Y.shape
            return -Sum(M[i, i1] * Y[i1, i2].diff(v) * M[i2, j], (i1, 0, r1 - 1), (i2, 0, r2 - 1))
        if self.has(v.args[0]):
            return None
        return S.Zero