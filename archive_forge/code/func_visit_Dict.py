from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_Dict(self, node):
    self._write('{')
    for key, value in zip(node.keys, node.values):
        self.visit(key)
        self._write(': ')
        self.visit(value)
        self._write(', ')
    self._write('}')