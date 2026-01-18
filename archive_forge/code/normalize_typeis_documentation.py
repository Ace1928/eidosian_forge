from pythran.passmanager import Transformation
import gast as ast
from functools import reduce


    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("""
    ... def foo(y):
    ...  return builtins.int == builtins.type(y)""")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(NormalizeTypeIs, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(y):
        return builtins.isinstance(y, builtins.int)
    