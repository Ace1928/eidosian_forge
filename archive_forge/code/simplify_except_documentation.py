from pythran.passmanager import Transformation
import gast as ast

    Remove redundant except clauses

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse('try: pass\nexcept (OSError, OSError): pass')
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(SimplifyExcept, node)
    >>> print(pm.dump(backend.Python, node))
    try:
        pass
    except OSError:
        pass
    