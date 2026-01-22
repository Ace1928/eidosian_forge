from collections import defaultdict
from operator import index as index_
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer, Rational
from sympy.core.sympify import _sympify, SympifyError
from sympy.core.singleton import S
from sympy.polys.domains import ZZ, QQ, EXRAW
from sympy.polys.matrices import DomainMatrix
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent
from .common import classof
from .matrices import MatrixBase, MatrixKind, ShapeError
def _getitem_RepMatrix(self, key):
    """Return portion of self defined by key. If the key involves a slice
    then a list will be returned (if key is a single slice) or a matrix
    (if key was a tuple involving a slice).

    Examples
    ========

    >>> from sympy import Matrix, I
    >>> m = Matrix([
    ... [1, 2 + I],
    ... [3, 4    ]])

    If the key is a tuple that does not involve a slice then that element
    is returned:

    >>> m[1, 0]
    3

    When a tuple key involves a slice, a matrix is returned. Here, the
    first column is selected (all rows, column 0):

    >>> m[:, 0]
    Matrix([
    [1],
    [3]])

    If the slice is not a tuple then it selects from the underlying
    list of elements that are arranged in row order and a list is
    returned if a slice is involved:

    >>> m[0]
    1
    >>> m[::2]
    [1, 3]
    """
    if isinstance(key, tuple):
        i, j = key
        try:
            return self._rep.getitem_sympy(index_(i), index_(j))
        except (TypeError, IndexError):
            if isinstance(i, Expr) and (not i.is_number) or (isinstance(j, Expr) and (not j.is_number)):
                if (j < 0) is True or (j >= self.shape[1]) is True or (i < 0) is True or ((i >= self.shape[0]) is True):
                    raise ValueError('index out of boundary')
                from sympy.matrices.expressions.matexpr import MatrixElement
                return MatrixElement(self, i, j)
            if isinstance(i, slice):
                i = range(self.rows)[i]
            elif is_sequence(i):
                pass
            else:
                i = [i]
            if isinstance(j, slice):
                j = range(self.cols)[j]
            elif is_sequence(j):
                pass
            else:
                j = [j]
            return self.extract(i, j)
    else:
        rows, cols = self.shape
        if not rows * cols:
            return [][key]
        rep = self._rep.rep
        domain = rep.domain
        is_slice = isinstance(key, slice)
        if is_slice:
            values = [rep.getitem(*divmod(n, cols)) for n in range(rows * cols)[key]]
        else:
            values = [rep.getitem(*divmod(index_(key), cols))]
        if domain != EXRAW:
            to_sympy = domain.to_sympy
            values = [to_sympy(val) for val in values]
        if is_slice:
            return values
        else:
            return values[0]