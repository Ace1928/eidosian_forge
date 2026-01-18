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
def transformation(self, sys):
    """
        Return coordinate transformation function from *self* to *sys*.

        Parameters
        ==========

        sys : CoordSystem

        Returns
        =======

        sympy.Lambda

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.transformation(R2_p)
        Lambda((x, y), Matrix([
        [sqrt(x**2 + y**2)],
        [      atan2(y, x)]]))

        """
    signature = self.args[2]
    key = Tuple(self.name, sys.name)
    if self == sys:
        expr = Matrix(self.symbols)
    elif key in self.relations:
        expr = Matrix(self.relations[key][1])
    elif key[::-1] in self.relations:
        expr = Matrix(self._inverse_transformation(sys, self))
    else:
        expr = Matrix(self._indirect_transformation(self, sys))
    return Lambda(signature, expr)