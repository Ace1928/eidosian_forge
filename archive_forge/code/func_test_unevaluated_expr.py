from symengine import (
from symengine.test_utilities import raises
import unittest
def test_unevaluated_expr():
    x = Symbol('x')
    t = UnevaluatedExpr(x)
    assert x + t != 2 * x
    assert not t.is_number
    assert not t.is_integer
    assert not t.is_finite
    t = UnevaluatedExpr(1)
    assert t.is_number
    assert t.is_integer
    assert t.is_finite