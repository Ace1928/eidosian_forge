from pythran.passmanager import Transformation
from pythran.utils import isnum
import gast as ast

    Turns iteration over an incrementing list of literals into a range

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("for i in [1,2,3]: print(i)")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(RangeLoopUnfolding, node)
    >>> print(pm.dump(backend.Python, node))
    for i in builtins.range(1, 4, 1):
        print(i)
    