from . import c_ast
def visit_DoWhile(self, n):
    s = 'do\n'
    s += self._generate_stmt(n.stmt, add_indent=True)
    s += self._make_indent() + 'while ('
    if n.cond:
        s += self.visit(n.cond)
    s += ');'
    return s