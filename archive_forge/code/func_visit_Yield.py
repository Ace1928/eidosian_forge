from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_Yield(self, node):
    self._write('yield')
    if getattr(node, 'value', None):
        self._write(' ')
        self.visit(node.value)