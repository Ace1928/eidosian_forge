from sympy.core.function import (Function, FunctionClass)
from sympy.core.symbol import (Symbol, var)
from sympy.testing.pytest import raises
raises(ValueError, lambda: var(''))