from pythran.analyses import GlobalDeclarations, ImportedIds
from pythran.analyses import Check
from pythran.analyses import ExtendedDefUseChains
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import mangle
import pythran.metadata as metadata
from copy import copy, deepcopy
import gast as ast

    Turns lambda into top-level functions.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(y): lambda x:y+x")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(RemoveLambdas, node)
    >>> print(pm.dump(backend.Python, node))
    import functools as __pythran_import_functools
    def foo(y):
        __pythran_import_functools.partial(foo_lambda0, y)
    def foo_lambda0(y, x):
        return (y + x)
    