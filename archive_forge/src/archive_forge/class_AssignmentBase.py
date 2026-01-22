from __future__ import annotations
from typing import Any
from collections import defaultdict
from sympy.core.relational import (Ge, Gt, Le, Lt)
from sympy.core import Symbol, Tuple, Dummy
from sympy.core.basic import Basic
from sympy.core.expr import Expr, Atom
from sympy.core.numbers import Float, Integer, oo
from sympy.core.sympify import _sympify, sympify, SympifyError
from sympy.utilities.iterables import (iterable, topological_sort,
class AssignmentBase(CodegenAST):
    """ Abstract base class for Assignment and AugmentedAssignment.

    Attributes:
    ===========

    op : str
        Symbol for assignment operator, e.g. "=", "+=", etc.
    """

    def __new__(cls, lhs, rhs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        cls._check_args(lhs, rhs)
        return super().__new__(cls, lhs, rhs)

    @property
    def lhs(self):
        return self.args[0]

    @property
    def rhs(self):
        return self.args[1]

    @classmethod
    def _check_args(cls, lhs, rhs):
        """ Check arguments to __new__ and raise exception if any problems found.

        Derived classes may wish to override this.
        """
        from sympy.matrices.expressions.matexpr import MatrixElement, MatrixSymbol
        from sympy.tensor.indexed import Indexed
        from sympy.tensor.array.expressions import ArrayElement
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Element, Variable, ArrayElement)
        if not isinstance(lhs, assignable):
            raise TypeError('Cannot assign to lhs of type %s.' % type(lhs))
        lhs_is_mat = hasattr(lhs, 'shape') and (not isinstance(lhs, Indexed))
        rhs_is_mat = hasattr(rhs, 'shape') and (not isinstance(rhs, Indexed))
        if lhs_is_mat:
            if not rhs_is_mat:
                raise ValueError('Cannot assign a scalar to a matrix.')
            elif lhs.shape != rhs.shape:
                raise ValueError('Dimensions of lhs and rhs do not align.')
        elif rhs_is_mat and (not lhs_is_mat):
            raise ValueError('Cannot assign a matrix to a scalar.')