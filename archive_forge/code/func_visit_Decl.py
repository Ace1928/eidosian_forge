from . import c_ast
def visit_Decl(self, n, no_type=False):
    s = n.name if no_type else self._generate_decl(n)
    if n.bitsize:
        s += ' : ' + self.visit(n.bitsize)
    if n.init:
        s += ' = ' + self._visit_expr(n.init)
    return s