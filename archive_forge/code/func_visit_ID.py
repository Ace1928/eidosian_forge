from . import c_ast
def visit_ID(self, n):
    return n.name