from pythran.analyses import Identifiers
from pythran.passmanager import Transformation
import gast as ast

    Prevents parameter shadowing by creating new variable.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(a): a = 1")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(UnshadowParameters, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(a):
        a_ = a
        a_ = 1
    