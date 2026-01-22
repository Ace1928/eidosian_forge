from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class SortRef(AstRef):
    """A Sort is essentially a type. Every Z3 expression has a sort. A sort is an AST node."""

    def as_ast(self):
        return Z3_sort_to_ast(self.ctx_ref(), self.ast)

    def get_id(self):
        return Z3_get_ast_id(self.ctx_ref(), self.as_ast())

    def kind(self):
        """Return the Z3 internal kind of a sort.
        This method can be used to test if `self` is one of the Z3 builtin sorts.

        >>> b = BoolSort()
        >>> b.kind() == Z3_BOOL_SORT
        True
        >>> b.kind() == Z3_INT_SORT
        False
        >>> A = ArraySort(IntSort(), IntSort())
        >>> A.kind() == Z3_ARRAY_SORT
        True
        >>> A.kind() == Z3_INT_SORT
        False
        """
        return _sort_kind(self.ctx, self.ast)

    def subsort(self, other):
        """Return `True` if `self` is a subsort of `other`.

        >>> IntSort().subsort(RealSort())
        True
        """
        return False

    def cast(self, val):
        """Try to cast `val` as an element of sort `self`.

        This method is used in Z3Py to convert Python objects such as integers,
        floats, longs and strings into Z3 expressions.

        >>> x = Int('x')
        >>> RealSort().cast(x)
        ToReal(x)
        """
        if z3_debug():
            _z3_assert(is_expr(val), 'Z3 expression expected')
            _z3_assert(self.eq(val.sort()), 'Sort mismatch')
        return val

    def name(self):
        """Return the name (string) of sort `self`.

        >>> BoolSort().name()
        'Bool'
        >>> ArraySort(IntSort(), IntSort()).name()
        'Array'
        """
        return _symbol2py(self.ctx, Z3_get_sort_name(self.ctx_ref(), self.ast))

    def __eq__(self, other):
        """Return `True` if `self` and `other` are the same Z3 sort.

        >>> p = Bool('p')
        >>> p.sort() == BoolSort()
        True
        >>> p.sort() == IntSort()
        False
        """
        if other is None:
            return False
        return Z3_is_eq_sort(self.ctx_ref(), self.ast, other.ast)

    def __ne__(self, other):
        """Return `True` if `self` and `other` are not the same Z3 sort.

        >>> p = Bool('p')
        >>> p.sort() != BoolSort()
        False
        >>> p.sort() != IntSort()
        True
        """
        return not Z3_is_eq_sort(self.ctx_ref(), self.ast, other.ast)

    def __hash__(self):
        """ Hash code. """
        return AstRef.__hash__(self)