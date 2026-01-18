from pythran.analyses import PotentialIterator
from pythran.passmanager import Transformation
import gast as ast

    Transforms list comprehension into genexp
    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("""                   \ndef foo(l):                                    \n    return builtins.sum(l)                     \ndef bar(n):                                    \n    return foo([x for x in builtins.range(n)]) """)
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(ListCompToGenexp, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(l):
        return builtins.sum(l)
    def bar(n):
        return foo((x for x in builtins.range(n)))
    