from . import c_ast
def visit_ExprList(self, n):
    visited_subexprs = []
    for expr in n.exprs:
        visited_subexprs.append(self._visit_expr(expr))
    return ', '.join(visited_subexprs)