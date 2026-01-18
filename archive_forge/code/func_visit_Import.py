from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_Import(self, node):
    self._new_line()
    self._write('import ')
    self.visit(node.names[0])
    for name in node.names[1:]:
        self._write(', ')
        self.visit(name)