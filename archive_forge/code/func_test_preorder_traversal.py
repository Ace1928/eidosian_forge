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
def test_preorder_traversal():
    expr = Basic(b21, b3)
    assert list(preorder_traversal(expr)) == [expr, b21, b2, b1, b1, b3, b2, b1]
    assert list(preorder_traversal(('abc', ('d', 'ef')))) == [('abc', ('d', 'ef')), 'abc', ('d', 'ef'), 'd', 'ef']
    result = []
    pt = preorder_traversal(expr)
    for i in pt:
        result.append(i)
        if i == b2:
            pt.skip()
    assert result == [expr, b21, b2, b1, b3, b2]
    w, x, y, z = symbols('w:z')
    expr = z + w * (x + y)
    assert list(preorder_traversal([expr], keys=default_sort_key)) == [[w * (x + y) + z], w * (x + y) + z, z, w * (x + y), w, x + y, x, y]
    assert list(preorder_traversal((x + y) * z, keys=True)) == [z * (x + y), z, x + y, x, y]