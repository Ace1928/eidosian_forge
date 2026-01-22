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
class _deprecated_container:

    def __init__(self, message, data):
        super().__init__(data)
        self.message = message

    def warn(self):
        sympy_deprecation_warning(self.message, deprecated_since_version='1.7', active_deprecations_target='deprecated-diffgeom-mutable', stacklevel=4)

    def __iter__(self):
        self.warn()
        return super().__iter__()

    def __getitem__(self, key):
        self.warn()
        return super().__getitem__(key)

    def __contains__(self, key):
        self.warn()
        return super().__contains__(key)