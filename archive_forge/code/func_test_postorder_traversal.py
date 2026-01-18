from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import symbols
from sympy.core.singleton import S
from sympy.core.function import expand, Function
from sympy.core.numbers import I
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import factor
from sympy.core.traversal import preorder_traversal, use, postorder_traversal, iterargs, iterfreeargs
from sympy.functions.elementary.piecewise import ExprCondPair, Piecewise
from sympy.testing.pytest import warns_deprecated_sympy
from sympy.utilities.iterables import capture
def test_postorder_traversal():
    x, y, z, w = symbols('x y z w')
    expr = z + w * (x + y)
    expected = [z, w, x, y, x + y, w * (x + y), w * (x + y) + z]
    assert list(postorder_traversal(expr, keys=default_sort_key)) == expected
    assert list(postorder_traversal(expr, keys=True)) == expected
    expr = Piecewise((x, x < 1), (x ** 2, True))
    expected = [x, 1, x, x < 1, ExprCondPair(x, x < 1), 2, x, x ** 2, S.true, ExprCondPair(x ** 2, True), Piecewise((x, x < 1), (x ** 2, True))]
    assert list(postorder_traversal(expr, keys=default_sort_key)) == expected
    assert list(postorder_traversal([expr], keys=default_sort_key)) == expected + [[expr]]
    assert list(postorder_traversal(Integral(x ** 2, (x, 0, 1)), keys=default_sort_key)) == [2, x, x ** 2, 0, 1, x, Tuple(x, 0, 1), Integral(x ** 2, Tuple(x, 0, 1))]
    assert list(postorder_traversal(('abc', ('d', 'ef')))) == ['abc', 'd', 'ef', ('d', 'ef'), ('abc', ('d', 'ef'))]