from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.logic.boolalg import And
from sympy.core.symbol import Str
from sympy.unify.core import Compound, Variable
from sympy.unify.usympy import (deconstruct, construct, unify, is_associative,
from sympy.abc import x, y, z, n
def test_unify_commutative():
    expr = Add(1, 2, 3, evaluate=False)
    a, b, c = map(Symbol, 'abc')
    pattern = Add(a, b, c, evaluate=False)
    result = tuple(unify(expr, pattern, {}, (a, b, c)))
    expected = ({a: 1, b: 2, c: 3}, {a: 1, b: 3, c: 2}, {a: 2, b: 1, c: 3}, {a: 2, b: 3, c: 1}, {a: 3, b: 1, c: 2}, {a: 3, b: 2, c: 1})
    assert iterdicteq(result, expected)