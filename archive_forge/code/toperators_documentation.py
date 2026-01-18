from sympy import permutedims
from sympy.core.numbers import Number
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.tensor.tensor import Tensor, TensExpr, TensAdd, TensMul

    Partial derivative for tensor expressions.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead
    >>> from sympy.tensor.toperators import PartialDerivative
    >>> from sympy import symbols
    >>> L = TensorIndexType("L")
    >>> A = TensorHead("A", [L])
    >>> B = TensorHead("B", [L])
    >>> i, j, k = symbols("i j k")

    >>> expr = PartialDerivative(A(i), A(j))
    >>> expr
    PartialDerivative(A(i), A(j))

    The ``PartialDerivative`` object behaves like a tensorial expression:

    >>> expr.get_indices()
    [i, -j]

    Notice that the deriving variables have opposite valence than the
    printed one: ``A(j)`` is printed as covariant, but the index of the
    derivative is actually contravariant, i.e. ``-j``.

    Indices can be contracted:

    >>> expr = PartialDerivative(A(i), A(i))
    >>> expr
    PartialDerivative(A(L_0), A(L_0))
    >>> expr.get_indices()
    [L_0, -L_0]

    The method ``.get_indices()`` always returns all indices (even the
    contracted ones). If only uncontracted indices are needed, call
    ``.get_free_indices()``:

    >>> expr.get_free_indices()
    []

    Nested partial derivatives are flattened:

    >>> expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(k))
    >>> expr
    PartialDerivative(A(i), A(j), A(k))
    >>> expr.get_indices()
    [i, -j, -k]

    Replace a derivative with array values:

    >>> from sympy.abc import x, y
    >>> from sympy import sin, log
    >>> compA = [sin(x), log(x)*y**3]
    >>> compB = [x, y]
    >>> expr = PartialDerivative(A(i), B(j))
    >>> expr.replace_with_arrays({A(i): compA, B(i): compB})
    [[cos(x), 0], [y**3/x, 3*y**2*log(x)]]

    The returned array is indexed by `(i, -j)`.

    Be careful that other SymPy modules put the indices of the deriving
    variables before the indices of the derivand in the derivative result.
    For example:

    >>> expr.get_free_indices()
    [i, -j]

    >>> from sympy import Matrix, Array
    >>> Matrix(compA).diff(Matrix(compB)).reshape(2, 2)
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]
    >>> Array(compA).diff(Array(compB))
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]

    These are the transpose of the result of ``PartialDerivative``,
    as the matrix and the array modules put the index `-j` before `i` in the
    derivative result. An array read with index order `(-j, i)` is indeed the
    transpose of the same array read with index order `(i, -j)`. By specifying
    the index order to ``.replace_with_arrays`` one can get a compatible
    expression:

    >>> expr.replace_with_arrays({A(i): compA, B(i): compB}, [-j, i])
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]
    