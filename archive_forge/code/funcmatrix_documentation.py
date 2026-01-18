from .matexpr import MatrixExpr
from sympy.core.function import FunctionClass, Lambda
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify, sympify
from sympy.matrices import Matrix
from sympy.functions.elementary.complexes import re, im
Represents a matrix using a function (``Lambda``) which gives
    outputs according to the coordinates of each matrix entries.

    Parameters
    ==========

    rows : nonnegative integer. Can be symbolic.

    cols : nonnegative integer. Can be symbolic.

    lamda : Function, Lambda or str
        If it is a SymPy ``Function`` or ``Lambda`` instance,
        it should be able to accept two arguments which represents the
        matrix coordinates.

        If it is a pure string containing Python ``lambda`` semantics,
        it is interpreted by the SymPy parser and casted into a SymPy
        ``Lambda`` instance.

    Examples
    ========

    Creating a ``FunctionMatrix`` from ``Lambda``:

    >>> from sympy import FunctionMatrix, symbols, Lambda, MatPow
    >>> i, j, n, m = symbols('i,j,n,m')
    >>> FunctionMatrix(n, m, Lambda((i, j), i + j))
    FunctionMatrix(n, m, Lambda((i, j), i + j))

    Creating a ``FunctionMatrix`` from a SymPy function:

    >>> from sympy import KroneckerDelta
    >>> X = FunctionMatrix(3, 3, KroneckerDelta)
    >>> X.as_explicit()
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    Creating a ``FunctionMatrix`` from a SymPy undefined function:

    >>> from sympy import Function
    >>> f = Function('f')
    >>> X = FunctionMatrix(3, 3, f)
    >>> X.as_explicit()
    Matrix([
    [f(0, 0), f(0, 1), f(0, 2)],
    [f(1, 0), f(1, 1), f(1, 2)],
    [f(2, 0), f(2, 1), f(2, 2)]])

    Creating a ``FunctionMatrix`` from Python ``lambda``:

    >>> FunctionMatrix(n, m, 'lambda i, j: i + j')
    FunctionMatrix(n, m, Lambda((i, j), i + j))

    Example of lazy evaluation of matrix product:

    >>> Y = FunctionMatrix(1000, 1000, Lambda((i, j), i + j))
    >>> isinstance(Y*Y, MatPow) # this is an expression object
    True
    >>> (Y**2)[10,10] # So this is evaluated lazily
    342923500

    Notes
    =====

    This class provides an alternative way to represent an extremely
    dense matrix with entries in some form of a sequence, in a most
    sparse way.
    