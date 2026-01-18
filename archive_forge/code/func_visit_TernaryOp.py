from . import c_ast
def visit_TernaryOp(self, n):
    s = '(' + self._visit_expr(n.cond) + ') ? '
    s += '(' + self._visit_expr(n.iftrue) + ') : '
    s += '(' + self._visit_expr(n.iffalse) + ')'
    return s