from pythran.analyses import DefUseChains
from pythran.passmanager import Transformation
import pythran.metadata as metadata

    Remove useless local functions

    >>> import gast as ast
    >>> from pythran import passmanager, backend, metadata
    >>> pm = passmanager.PassManager("test")
    >>> node = ast.parse("def foo(): return 1")
    >>> _, node = pm.apply(RemoveDeadFunctions, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        return 1
    >>> node = ast.parse("def foo(): return 1")
    >>> metadata.add(node.body[0], metadata.Local())
    >>> _, node = pm.apply(RemoveDeadFunctions, node)
    >>> print(pm.dump(backend.Python, node))
    <BLANKLINE>
    