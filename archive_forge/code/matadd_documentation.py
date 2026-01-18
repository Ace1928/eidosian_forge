from functools import reduce
import operator
from sympy.core import Basic, sympify
from sympy.core.add import add, Add, _could_extract_minus_sign
from sympy.core.sorting import default_sort_key
from sympy.functions import adjoint
from sympy.matrices.matrices import MatrixBase
from sympy.matrices.expressions.transpose import transpose
from sympy.strategies import (rm_id, unpack, flatten, sort, condition,
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix, GenericZeroMatrix
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
from sympy.utilities.iterables import sift
from sympy.utilities.exceptions import sympy_deprecation_warning
 Merge explicit MatrixBase arguments

    Examples
    ========

    >>> from sympy import MatrixSymbol, eye, Matrix, MatAdd, pprint
    >>> from sympy.matrices.expressions.matadd import merge_explicit
    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = eye(2)
    >>> C = Matrix([[1, 2], [3, 4]])
    >>> X = MatAdd(A, B, C)
    >>> pprint(X)
        [1  0]   [1  2]
    A + [    ] + [    ]
        [0  1]   [3  4]
    >>> pprint(merge_explicit(X))
        [2  2]
    A + [    ]
        [3  5]
    