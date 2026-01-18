import gast as ast
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
Turns f-strings to format syntax with modulus

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("f'a = {1+1:4d}; b = {b:s};'")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(RemoveFStrings, node)
    >>> print(pm.dump(backend.Python, node))
    ('a = %4d; b = %s;' % ((1 + 1), b))
    