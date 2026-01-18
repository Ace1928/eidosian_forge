from sympy.core.function import (Function, FunctionClass)
from sympy.core.symbol import (Symbol, var)
from sympy.testing.pytest import raises
def test_var_accepts_comma():
    ns = {'var': var}
    v1 = eval("var('x y z')", ns)
    v2 = eval("var('x,y,z')", ns)
    v3 = eval("var('x,y z')", ns)
    assert v1 == v2
    assert v1 == v3