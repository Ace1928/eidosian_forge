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
class _MinimalMatrix:
    """Class providing the minimum functionality
    for a matrix-like object and implementing every method
    required for a `MatrixRequired`.  This class does not have everything
    needed to become a full-fledged SymPy object, but it will satisfy the
    requirements of anything inheriting from `MatrixRequired`.  If you wish
    to make a specialized matrix type, make sure to implement these
    methods and properties with the exception of `__init__` and `__repr__`
    which are included for convenience."""
    is_MatrixLike = True
    _sympify = staticmethod(sympify)
    _class_priority = 3
    zero = S.Zero
    one = S.One
    is_Matrix = True
    is_MatrixExpr = False

    @classmethod
    def _new(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __init__(self, rows, cols=None, mat=None, copy=False):
        if isfunction(mat):
            mat = [mat(i, j) for i in range(rows) for j in range(cols)]
        if cols is None and mat is None:
            mat = rows
        rows, cols = getattr(mat, 'shape', (rows, cols))
        try:
            if cols is None and mat is None:
                mat = rows
            cols = len(mat[0])
            rows = len(mat)
            mat = [x for l in mat for x in l]
        except (IndexError, TypeError):
            pass
        self.mat = tuple((self._sympify(x) for x in mat))
        self.rows, self.cols = (rows, cols)
        if self.rows is None or self.cols is None:
            raise NotImplementedError('Cannot initialize matrix with given parameters')

    def __getitem__(self, key):

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

        def _coord_to_index(i, j):
            """Return the index in _mat corresponding
            to the (i,j) position in the matrix. """
            return i * self.cols + j
        if isinstance(key, tuple):
            i, j = key
            if isinstance(i, slice) or isinstance(j, slice):
                i, j = _normalize_slices(i, j)
                rowsList, colsList = (list(range(self.rows))[i], list(range(self.cols))[j])
                indices = (i * self.cols + j for i in rowsList for j in colsList)
                return self._new(len(rowsList), len(colsList), [self.mat[i] for i in indices])
            key = _coord_to_index(i, j)
        return self.mat[key]

    def __eq__(self, other):
        try:
            classof(self, other)
        except TypeError:
            return False
        return self.shape == other.shape and list(self) == list(other)

    def __len__(self):
        return self.rows * self.cols

    def __repr__(self):
        return '_MinimalMatrix({}, {}, {})'.format(self.rows, self.cols, self.mat)

    @property
    def shape(self):
        return (self.rows, self.cols)