from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.symbol import (Symbol, symbols)
from sympy.testing.pytest import XFAIL
@XFAIL
def test_subexpression_postprocessors():
    a = symbols('a')
    x = SymbolInMulOnce('x')
    w = SymbolRemovesOtherSymbols('w')
    assert 3 * a * w ** 2 == 3 * w ** 2
    assert 3 * a * x ** 3 * w ** 2 == 3 * w ** 2
    x = SubclassSymbolInMulOnce('x')
    w = SubclassSymbolRemovesOtherSymbols('w')
    assert 3 * a * w ** 2 == 3 * w ** 2
    assert 3 * a * x ** 3 * w ** 2 == 3 * w ** 2