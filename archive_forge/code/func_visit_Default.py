from . import c_ast
def visit_Default(self, n):
    s = 'default:\n'
    for stmt in n.stmts:
        s += self._generate_stmt(stmt, add_indent=True)
    return s