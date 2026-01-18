import pytest
from numpy.f2py.symbolic import (
from . import util
def test_tostring_fortran(self):
    x = as_symbol('x')
    y = as_symbol('y')
    z = as_symbol('z')
    n = as_number(123)
    m = as_number(456)
    a = as_array((n, m))
    c = as_complex(n, m)
    assert str(x) == 'x'
    assert str(n) == '123'
    assert str(a) == '[123, 456]'
    assert str(c) == '(123, 456)'
    assert str(Expr(Op.TERMS, {x: 1})) == 'x'
    assert str(Expr(Op.TERMS, {x: 2})) == '2 * x'
    assert str(Expr(Op.TERMS, {x: -1})) == '-x'
    assert str(Expr(Op.TERMS, {x: -2})) == '-2 * x'
    assert str(Expr(Op.TERMS, {x: 1, y: 1})) == 'x + y'
    assert str(Expr(Op.TERMS, {x: -1, y: -1})) == '-x - y'
    assert str(Expr(Op.TERMS, {x: 2, y: 3})) == '2 * x + 3 * y'
    assert str(Expr(Op.TERMS, {x: -2, y: 3})) == '-2 * x + 3 * y'
    assert str(Expr(Op.TERMS, {x: 2, y: -3})) == '2 * x - 3 * y'
    assert str(Expr(Op.FACTORS, {x: 1})) == 'x'
    assert str(Expr(Op.FACTORS, {x: 2})) == 'x ** 2'
    assert str(Expr(Op.FACTORS, {x: -1})) == 'x ** -1'
    assert str(Expr(Op.FACTORS, {x: -2})) == 'x ** -2'
    assert str(Expr(Op.FACTORS, {x: 1, y: 1})) == 'x * y'
    assert str(Expr(Op.FACTORS, {x: 2, y: 3})) == 'x ** 2 * y ** 3'
    v = Expr(Op.FACTORS, {x: 2, Expr(Op.TERMS, {x: 1, y: 1}): 3})
    assert str(v) == 'x ** 2 * (x + y) ** 3', str(v)
    v = Expr(Op.FACTORS, {x: 2, Expr(Op.FACTORS, {x: 1, y: 1}): 3})
    assert str(v) == 'x ** 2 * (x * y) ** 3', str(v)
    assert str(Expr(Op.APPLY, ('f', (), {}))) == 'f()'
    assert str(Expr(Op.APPLY, ('f', (x,), {}))) == 'f(x)'
    assert str(Expr(Op.APPLY, ('f', (x, y), {}))) == 'f(x, y)'
    assert str(Expr(Op.INDEXING, ('f', x))) == 'f[x]'
    assert str(as_ternary(x, y, z)) == 'merge(y, z, x)'
    assert str(as_eq(x, y)) == 'x .eq. y'
    assert str(as_ne(x, y)) == 'x .ne. y'
    assert str(as_lt(x, y)) == 'x .lt. y'
    assert str(as_le(x, y)) == 'x .le. y'
    assert str(as_gt(x, y)) == 'x .gt. y'
    assert str(as_ge(x, y)) == 'x .ge. y'