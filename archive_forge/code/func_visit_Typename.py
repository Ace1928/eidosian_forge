from . import c_ast
def visit_Typename(self, n):
    return self._generate_type(n.type)