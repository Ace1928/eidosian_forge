from pythran.analyses import LocalNameDeclarations
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.utils import path_to_attr
from pythran import metadata
import gast as ast
class GlobalTransformer(ast.NodeTransformer):
    """
    Use assumptions on globals to improve code generation
    """

    def visit_Call(self, node):
        return node

    def visit_List(self, node):
        return ast.Call(path_to_attr(('builtins', 'pythran', 'static_list')), [ast.Tuple([self.visit(elt) for elt in node.elts], ast.Load())], [])