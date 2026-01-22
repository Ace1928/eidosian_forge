from collections import defaultdict
from collections.abc import Iterable
from inspect import isfunction
from functools import reduce
from sympy.assumptions.refine import refine
from sympy.core import SympifyError, Add
from sympy.core.basic import Atom
from sympy.core.decorators import call_highest_priority
from sympy.core.kind import Kind, NumberKind
from sympy.core.logic import fuzzy_and, FuzzyBool
from sympy.core.mod import Mod
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs, re, im
from .utilities import _dotprodsimp, _simplify
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.misc import as_int, filldedent
from sympy.tensor.array import NDimArray
from .utilities import _get_intermediate_simp_bool
def _normalize_slices(row_slice, col_slice):
    """Ensure that row_slice and col_slice do not have
            `None` in their arguments.  Any integers are converted
            to slices of length 1"""
    if not isinstance(row_slice, slice):
        row_slice = slice(row_slice, row_slice + 1, None)
    row_slice = slice(*row_slice.indices(self.rows))
    if not isinstance(col_slice, slice):
        col_slice = slice(col_slice, col_slice + 1, None)
    col_slice = slice(*col_slice.indices(self.cols))
    return (row_slice, col_slice)