from . import c_ast
def visit_ParamList(self, n):
    return ', '.join((self.visit(param) for param in n.params))