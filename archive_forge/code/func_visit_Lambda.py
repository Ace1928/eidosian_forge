from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
@with_parens
def visit_Lambda(self, node):
    self._write('lambda ')
    self.visit(node.args)
    self._write(': ')
    self.visit(node.body)