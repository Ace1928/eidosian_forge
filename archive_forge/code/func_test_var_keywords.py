from sympy.core.function import (Function, FunctionClass)
from sympy.core.symbol import (Symbol, var)
from sympy.testing.pytest import raises
def test_var_keywords():
    ns = {'var': var}
    eval("var('x y', real=True)", ns)
    assert ns['x'].is_real and ns['y'].is_real