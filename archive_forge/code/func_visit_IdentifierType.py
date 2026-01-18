from . import c_ast
def visit_IdentifierType(self, n):
    return ' '.join(n.names)