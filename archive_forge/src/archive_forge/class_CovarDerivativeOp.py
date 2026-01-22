from __future__ import annotations
from typing import Any
from functools import reduce
from itertools import permutations
from sympy.combinatorics import Permutation
from sympy.core import (
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol, Dummy
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions import factorial
from sympy.matrices import ImmutableDenseMatrix as Matrix
from sympy.solvers import solve
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.simplify.simplify import simplify
class CovarDerivativeOp(Expr):
    """Covariant derivative operator.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import CovarDerivativeOp
    >>> from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct
    >>> TP = TensorProduct
    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()
    >>> ch = metric_to_Christoffel_2nd(TP(dx, dx) + TP(dy, dy))

    >>> ch
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> cvd = CovarDerivativeOp(fx*e_x, ch)
    >>> cvd(fx)
    x
    >>> cvd(fx*e_x)
    x*e_x

    """

    def __new__(cls, wrt, christoffel):
        if len({v._coord_sys for v in wrt.atoms(BaseVectorField)}) > 1:
            raise NotImplementedError()
        if contravariant_order(wrt) != 1 or covariant_order(wrt):
            raise ValueError('Covariant derivatives are defined only with respect to vector fields. The supplied argument was not a vector field.')
        christoffel = ImmutableDenseNDimArray(christoffel)
        obj = super().__new__(cls, wrt, christoffel)
        obj._wrt = wrt
        obj._christoffel = christoffel
        return obj

    @property
    def wrt(self):
        return self.args[0]

    @property
    def christoffel(self):
        return self.args[1]

    def __call__(self, field):
        vectors = list(self._wrt.atoms(BaseVectorField))
        base_ops = [BaseCovarDerivativeOp(v._coord_sys, v._index, self._christoffel) for v in vectors]
        return self._wrt.subs(list(zip(vectors, base_ops))).rcall(field)