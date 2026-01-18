import pytest
from numpy.f2py.symbolic import (
from . import util
def test_tostring_c(self):
    language = Language.C
    x = as_symbol('x')
    y = as_symbol('y')
    z = as_symbol('z')
    n = as_number(123)
    assert Expr(Op.FACTORS, {x: 2}).tostring(language=language) == 'x * x'
    assert Expr(Op.FACTORS, {x + y: 2}).tostring(language=language) == '(x + y) * (x + y)'
    assert Expr(Op.FACTORS, {x: 12}).tostring(language=language) == 'pow(x, 12)'
    assert as_apply(ArithOp.DIV, x, y).tostring(language=language) == 'x / y'
    assert as_apply(ArithOp.DIV, x, x + y).tostring(language=language) == 'x / (x + y)'
    assert as_apply(ArithOp.DIV, x - y, x + y).tostring(language=language) == '(x - y) / (x + y)'
    assert (x + (x - y) / (x + y) + n).tostring(language=language) == '123 + x + (x - y) / (x + y)'
    assert as_ternary(x, y, z).tostring(language=language) == '(x?y:z)'
    assert as_eq(x, y).tostring(language=language) == 'x == y'
    assert as_ne(x, y).tostring(language=language) == 'x != y'
    assert as_lt(x, y).tostring(language=language) == 'x < y'
    assert as_le(x, y).tostring(language=language) == 'x <= y'
    assert as_gt(x, y).tostring(language=language) == 'x > y'
    assert as_ge(x, y).tostring(language=language) == 'x >= y'