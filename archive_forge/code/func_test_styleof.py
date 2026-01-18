from sympy.printing.dot import (purestr, styleof, attrprint, dotnode,
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import (Float, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing.repr import srepr
from sympy.abc import x
def test_styleof():
    styles = [(Basic, {'color': 'blue', 'shape': 'ellipse'}), (Expr, {'color': 'black'})]
    assert styleof(Basic(S(1)), styles) == {'color': 'blue', 'shape': 'ellipse'}
    assert styleof(x + 1, styles) == {'color': 'black', 'shape': 'ellipse'}