from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_Expression(self, node):
    self._new_line()
    return self.visit(node.body)