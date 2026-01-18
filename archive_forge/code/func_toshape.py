from pythran.analyses import Aliases
from pythran.tables import MODULES
from pythran.passmanager import Transformation
from pythran.utils import pythran_builtin_attr
import gast as ast
def toshape(node):
    b = pythran_builtin_attr('make_shape')
    return ast.Call(b, node.elts, [])