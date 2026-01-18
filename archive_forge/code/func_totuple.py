from pythran.analyses import Aliases, FixedSizeList
from pythran.tables import MODULES
from pythran.passmanager import Transformation
from pythran.utils import path_to_attr
import gast as ast
def totuple(node):
    return ast.Tuple(node.elts, node.ctx)