from sympy.core.containers import Dict
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int, filldedent
from .sparse import MutableSparseMatrix as SparseMatrix
Returns a SparseMatrix from the given dictionary describing
    the diagonals of the matrix. The keys are positive for upper
    diagonals and negative for those below the main diagonal. The
    values may be:

    * expressions or single-argument functions,

    * lists or tuples of values,

    * matrices

    Unless dimensions are given, the size of the returned matrix will
    be large enough to contain the largest non-zero value provided.

    kwargs
    ======

    rows : rows of the resulting matrix; computed if
           not given.

    cols : columns of the resulting matrix; computed if
           not given.

    Examples
    ========

    >>> from sympy import banded, ones, Matrix
    >>> from sympy.abc import x

    If explicit values are given in tuples,
    the matrix will autosize to contain all values, otherwise
    a single value is filled onto the entire diagonal:

    >>> banded({1: (1, 2, 3), -1: (4, 5, 6), 0: x})
    Matrix([
    [x, 1, 0, 0],
    [4, x, 2, 0],
    [0, 5, x, 3],
    [0, 0, 6, x]])

    A function accepting a single argument can be used to fill the
    diagonal as a function of diagonal index (which starts at 0).
    The size (or shape) of the matrix must be given to obtain more
    than a 1x1 matrix:

    >>> s = lambda d: (1 + d)**2
    >>> banded(5, {0: s, 2: s, -2: 2})
    Matrix([
    [1, 0, 1,  0,  0],
    [0, 4, 0,  4,  0],
    [2, 0, 9,  0,  9],
    [0, 2, 0, 16,  0],
    [0, 0, 2,  0, 25]])

    The diagonal of matrices placed on a diagonal will coincide
    with the indicated diagonal:

    >>> vert = Matrix([1, 2, 3])
    >>> banded({0: vert}, cols=3)
    Matrix([
    [1, 0, 0],
    [2, 1, 0],
    [3, 2, 1],
    [0, 3, 2],
    [0, 0, 3]])

    >>> banded(4, {0: ones(2)})
    Matrix([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1]])

    Errors are raised if the designated size will not hold
    all values an integral number of times. Here, the rows
    are designated as odd (but an even number is required to
    hold the off-diagonal 2x2 ones):

    >>> banded({0: 2, 1: ones(2)}, rows=5)
    Traceback (most recent call last):
    ...
    ValueError:
    sequence does not fit an integral number of times in the matrix

    And here, an even number of rows is given...but the square
    matrix has an even number of columns, too. As we saw
    in the previous example, an odd number is required:

    >>> banded(4, {0: 2, 1: ones(2)})  # trying to make 4x4 and cols must be odd
    Traceback (most recent call last):
    ...
    ValueError:
    sequence does not fit an integral number of times in the matrix

    A way around having to count rows is to enclosing matrix elements
    in a tuple and indicate the desired number of them to the right:

    >>> banded({0: 2, 2: (ones(2),)*3})
    Matrix([
    [2, 0, 1, 1, 0, 0, 0, 0],
    [0, 2, 1, 1, 0, 0, 0, 0],
    [0, 0, 2, 0, 1, 1, 0, 0],
    [0, 0, 0, 2, 1, 1, 0, 0],
    [0, 0, 0, 0, 2, 0, 1, 1],
    [0, 0, 0, 0, 0, 2, 1, 1]])

    An error will be raised if more than one value
    is written to a given entry. Here, the ones overlap
    with the main diagonal if they are placed on the
    first diagonal:

    >>> banded({0: (2,)*5, 1: (ones(2),)*3})
    Traceback (most recent call last):
    ...
    ValueError: collision at (1, 1)

    By placing a 0 at the bottom left of the 2x2 matrix of
    ones, the collision is avoided:

    >>> u2 = Matrix([
    ... [1, 1],
    ... [0, 1]])
    >>> banded({0: [2]*5, 1: [u2]*3})
    Matrix([
    [2, 1, 1, 0, 0, 0, 0],
    [0, 2, 1, 0, 0, 0, 0],
    [0, 0, 2, 1, 1, 0, 0],
    [0, 0, 0, 2, 1, 0, 0],
    [0, 0, 0, 0, 2, 1, 1],
    [0, 0, 0, 0, 0, 0, 1]])
    