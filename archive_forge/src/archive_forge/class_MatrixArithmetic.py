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
class MatrixArithmetic(MatrixRequired):
    """Provides basic matrix arithmetic operations.
    Should not be instantiated directly."""
    _op_priority = 10.01

    def _eval_Abs(self):
        return self._new(self.rows, self.cols, lambda i, j: Abs(self[i, j]))

    def _eval_add(self, other):
        return self._new(self.rows, self.cols, lambda i, j: self[i, j] + other[i, j])

    def _eval_matrix_mul(self, other):

        def entry(i, j):
            vec = [self[i, k] * other[k, j] for k in range(self.cols)]
            try:
                return Add(*vec)
            except (TypeError, SympifyError):
                return reduce(lambda a, b: a + b, vec)
        return self._new(self.rows, other.cols, entry)

    def _eval_matrix_mul_elementwise(self, other):
        return self._new(self.rows, self.cols, lambda i, j: self[i, j] * other[i, j])

    def _eval_matrix_rmul(self, other):

        def entry(i, j):
            return sum((other[i, k] * self[k, j] for k in range(other.cols)))
        return self._new(other.rows, self.cols, entry)

    def _eval_pow_by_recursion(self, num):
        if num == 1:
            return self
        if num % 2 == 1:
            a, b = (self, self._eval_pow_by_recursion(num - 1))
        else:
            a = b = self._eval_pow_by_recursion(num // 2)
        return a.multiply(b)

    def _eval_pow_by_cayley(self, exp):
        from sympy.discrete.recurrences import linrec_coeffs
        row = self.shape[0]
        p = self.charpoly()
        coeffs = (-p).all_coeffs()[1:]
        coeffs = linrec_coeffs(coeffs, exp)
        new_mat = self.eye(row)
        ans = self.zeros(row)
        for i in range(row):
            ans += coeffs[i] * new_mat
            new_mat *= self
        return ans

    def _eval_pow_by_recursion_dotprodsimp(self, num, prevsimp=None):
        if prevsimp is None:
            prevsimp = [True] * len(self)
        if num == 1:
            return self
        if num % 2 == 1:
            a, b = (self, self._eval_pow_by_recursion_dotprodsimp(num - 1, prevsimp=prevsimp))
        else:
            a = b = self._eval_pow_by_recursion_dotprodsimp(num // 2, prevsimp=prevsimp)
        m = a.multiply(b, dotprodsimp=False)
        lenm = len(m)
        elems = [None] * lenm
        for i in range(lenm):
            if prevsimp[i]:
                elems[i], prevsimp[i] = _dotprodsimp(m[i], withsimp=True)
            else:
                elems[i] = m[i]
        return m._new(m.rows, m.cols, elems)

    def _eval_scalar_mul(self, other):
        return self._new(self.rows, self.cols, lambda i, j: self[i, j] * other)

    def _eval_scalar_rmul(self, other):
        return self._new(self.rows, self.cols, lambda i, j: other * self[i, j])

    def _eval_Mod(self, other):
        return self._new(self.rows, self.cols, lambda i, j: Mod(self[i, j], other))

    def __abs__(self):
        """Returns a new matrix with entry-wise absolute values."""
        return self._eval_Abs()

    @call_highest_priority('__radd__')
    def __add__(self, other):
        """Return self + other, raising ShapeError if shapes do not match."""
        if isinstance(other, NDimArray):
            return NotImplemented
        other = _matrixify(other)
        if hasattr(other, 'shape'):
            if self.shape != other.shape:
                raise ShapeError('Matrix size mismatch: %s + %s' % (self.shape, other.shape))
        if getattr(other, 'is_Matrix', False):
            a, b = (self, other)
            if a.__class__ != classof(a, b):
                b, a = (a, b)
            return a._eval_add(b)
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_add(self, other)
        raise TypeError('cannot add %s and %s' % (type(self), type(other)))

    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self * (self.one / other)

    @call_highest_priority('__rmatmul__')
    def __matmul__(self, other):
        other = _matrixify(other)
        if not getattr(other, 'is_Matrix', False) and (not getattr(other, 'is_MatrixLike', False)):
            return NotImplemented
        return self.__mul__(other)

    def __mod__(self, other):
        return self.applyfunc(lambda x: x % other)

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        """Return self*other where other is either a scalar or a matrix
        of compatible dimensions.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> 2*A == A*2 == Matrix([[2, 4, 6], [8, 10, 12]])
        True
        >>> B = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> A*B
        Matrix([
        [30, 36, 42],
        [66, 81, 96]])
        >>> B*A
        Traceback (most recent call last):
        ...
        ShapeError: Matrices size mismatch.
        >>>

        See Also
        ========

        matrix_multiply_elementwise
        """
        return self.multiply(other)

    def multiply(self, other, dotprodsimp=None):
        """Same as __mul__() but with optional simplification.

        Parameters
        ==========

        dotprodsimp : bool, optional
            Specifies whether intermediate term algebraic simplification is used
            during matrix multiplications to control expression blowup and thus
            speed up calculation. Default is off.
        """
        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)
        other = _matrixify(other)
        if hasattr(other, 'shape') and len(other.shape) == 2 and (getattr(other, 'is_Matrix', True) or getattr(other, 'is_MatrixLike', True)):
            if self.shape[1] != other.shape[0]:
                raise ShapeError('Matrix size mismatch: %s * %s.' % (self.shape, other.shape))
        if getattr(other, 'is_Matrix', False):
            m = self._eval_matrix_mul(other)
            if isimpbool:
                return m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])
            return m
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_matrix_mul(self, other)
        if not isinstance(other, Iterable):
            try:
                return self._eval_scalar_mul(other)
            except TypeError:
                pass
        return NotImplemented

    def multiply_elementwise(self, other):
        """Return the Hadamard product (elementwise product) of A and B

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[0, 1, 2], [3, 4, 5]])
        >>> B = Matrix([[1, 10, 100], [100, 10, 1]])
        >>> A.multiply_elementwise(B)
        Matrix([
        [  0, 10, 200],
        [300, 40,   5]])

        See Also
        ========

        sympy.matrices.matrices.MatrixBase.cross
        sympy.matrices.matrices.MatrixBase.dot
        multiply
        """
        if self.shape != other.shape:
            raise ShapeError('Matrix shapes must agree {} != {}'.format(self.shape, other.shape))
        return self._eval_matrix_mul_elementwise(other)

    def __neg__(self):
        return self._eval_scalar_mul(-1)

    @call_highest_priority('__rpow__')
    def __pow__(self, exp):
        """Return self**exp a scalar or symbol."""
        return self.pow(exp)

    def pow(self, exp, method=None):
        """Return self**exp a scalar or symbol.

        Parameters
        ==========

        method : multiply, mulsimp, jordan, cayley
            If multiply then it returns exponentiation using recursion.
            If jordan then Jordan form exponentiation will be used.
            If cayley then the exponentiation is done using Cayley-Hamilton
            theorem.
            If mulsimp then the exponentiation is done using recursion
            with dotprodsimp. This specifies whether intermediate term
            algebraic simplification is used during naive matrix power to
            control expression blowup and thus speed up calculation.
            If None, then it heuristically decides which method to use.

        """
        if method is not None and method not in ['multiply', 'mulsimp', 'jordan', 'cayley']:
            raise TypeError('No such method')
        if self.rows != self.cols:
            raise NonSquareMatrixError()
        a = self
        jordan_pow = getattr(a, '_matrix_pow_by_jordan_blocks', None)
        exp = sympify(exp)
        if exp.is_zero:
            return a._new(a.rows, a.cols, lambda i, j: int(i == j))
        if exp == 1:
            return a
        diagonal = getattr(a, 'is_diagonal', None)
        if diagonal is not None and diagonal():
            return a._new(a.rows, a.cols, lambda i, j: a[i, j] ** exp if i == j else 0)
        if exp.is_Number and exp % 1 == 0:
            if a.rows == 1:
                return a._new([[a[0] ** exp]])
            if exp < 0:
                exp = -exp
                a = a.inv()
        if method == 'jordan':
            try:
                return jordan_pow(exp)
            except MatrixError:
                if method == 'jordan':
                    raise
        elif method == 'cayley':
            if not exp.is_Number or exp % 1 != 0:
                raise ValueError('cayley method is only valid for integer powers')
            return a._eval_pow_by_cayley(exp)
        elif method == 'mulsimp':
            if not exp.is_Number or exp % 1 != 0:
                raise ValueError('mulsimp method is only valid for integer powers')
            return a._eval_pow_by_recursion_dotprodsimp(exp)
        elif method == 'multiply':
            if not exp.is_Number or exp % 1 != 0:
                raise ValueError('multiply method is only valid for integer powers')
            return a._eval_pow_by_recursion(exp)
        elif method is None and exp.is_Number and (exp % 1 == 0):
            if a.rows == 2 and exp > 100000:
                return jordan_pow(exp)
            elif _get_intermediate_simp_bool(True, None):
                return a._eval_pow_by_recursion_dotprodsimp(exp)
            elif exp > 10000:
                return a._eval_pow_by_cayley(exp)
            else:
                return a._eval_pow_by_recursion(exp)
        if jordan_pow:
            try:
                return jordan_pow(exp)
            except NonInvertibleMatrixError:
                if exp.is_integer is False or exp.is_nonnegative is False:
                    raise
        from sympy.matrices.expressions import MatPow
        return MatPow(a, exp)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self + other

    @call_highest_priority('__matmul__')
    def __rmatmul__(self, other):
        other = _matrixify(other)
        if not getattr(other, 'is_Matrix', False) and (not getattr(other, 'is_MatrixLike', False)):
            return NotImplemented
        return self.__rmul__(other)

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self.rmultiply(other)

    def rmultiply(self, other, dotprodsimp=None):
        """Same as __rmul__() but with optional simplification.

        Parameters
        ==========

        dotprodsimp : bool, optional
            Specifies whether intermediate term algebraic simplification is used
            during matrix multiplications to control expression blowup and thus
            speed up calculation. Default is off.
        """
        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)
        other = _matrixify(other)
        if hasattr(other, 'shape') and len(other.shape) == 2 and (getattr(other, 'is_Matrix', True) or getattr(other, 'is_MatrixLike', True)):
            if self.shape[0] != other.shape[1]:
                raise ShapeError('Matrix size mismatch.')
        if getattr(other, 'is_Matrix', False):
            m = self._eval_matrix_rmul(other)
            if isimpbool:
                return m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])
            return m
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_matrix_rmul(self, other)
        if not isinstance(other, Iterable):
            try:
                return self._eval_scalar_rmul(other)
            except TypeError:
                pass
        return NotImplemented

    @call_highest_priority('__sub__')
    def __rsub__(self, a):
        return -self + a

    @call_highest_priority('__rsub__')
    def __sub__(self, a):
        return self + -a