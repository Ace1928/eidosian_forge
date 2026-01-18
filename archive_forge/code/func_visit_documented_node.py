from pythran.passmanager import Transformation
from pythran.utils import isstr
import gast as ast
def visit_documented_node(self, key, node):
    if node.body:
        first_stmt = node.body[0]
        if isinstance(first_stmt, ast.Expr):
            if isstr(first_stmt.value):
                self.update = True
                docstring = first_stmt.value.value
                self.docstrings[key] = docstring
                node.body.pop(0)
    return self.generic_visit(node)