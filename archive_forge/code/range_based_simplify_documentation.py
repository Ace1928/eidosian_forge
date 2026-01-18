from pythran.analyses import RangeValues
from pythran.passmanager import Transformation
import gast as ast
from math import isinf
from copy import deepcopy

    Simplify expressions based on range analysis

    >>> import gast as ast
    >>> from pythran import passmanager, backend

    >>> node = ast.parse("def any():\n for x in builtins.range(10): y=x%8")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(RangeBasedSimplify, node)
    >>> print(pm.dump(backend.Python, node))
    def any():
        for x in builtins.range(10):
            y = (x if (x < 8) else (x - 8))

    >>> node = ast.parse("def any(): x = 1 or 2; return 3 == x")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(RangeBasedSimplify, node)
    >>> print(pm.dump(backend.Python, node))
    def any():
        x = (1 or 2)
        return 0

    >>> node = ast.parse("def a(i): x = 1,1,2; return x[2], x[0 if i else 1]")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(RangeBasedSimplify, node)
    >>> print(pm.dump(backend.Python, node))
    def a(i):
        x = (1, 1, 2)
        return (2, 1)
    