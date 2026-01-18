from pythran.analyses import GlobalDeclarations, ImportedIds
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import mangle
import pythran.metadata as metadata
import gast as ast

    Replace nested function by top-level functions.

    Also add a call to a bind intrinsic that
    generates a local function with some arguments binded.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(x):\n def bar(y): return x+y\n bar(12)")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(RemoveNestedFunctions, node)
    >>> print(pm.dump(backend.Python, node))
    import functools as __pythran_import_functools
    def foo(x):
        bar = __pythran_import_functools.partial(pythran_bar0, x)
        bar(12)
    def pythran_bar0(x, y):
        return (x + y)
    