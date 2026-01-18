from . import c_ast
def visit_StaticAssert(self, n):
    s = '_Static_assert('
    s += self.visit(n.cond)
    if n.message:
        s += ','
        s += self.visit(n.message)
    s += ')'
    return s