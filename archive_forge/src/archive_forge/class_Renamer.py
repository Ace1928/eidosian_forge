from pythran.analyses import GlobalDeclarations, ImportedIds
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import mangle
import pythran.metadata as metadata
import gast as ast
class Renamer(ast.NodeTransformer):

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == former_name:
            node.func.id = new_name
            node.args = [ast.Name(iin, ast.Load(), None, None) for iin in sorted(ii)] + node.args
        return node