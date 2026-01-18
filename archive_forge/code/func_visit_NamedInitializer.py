from . import c_ast
def visit_NamedInitializer(self, n):
    s = ''
    for name in n.name:
        if isinstance(name, c_ast.ID):
            s += '.' + name.name
        else:
            s += '[' + self.visit(name) + ']'
    s += ' = ' + self._visit_expr(n.expr)
    return s