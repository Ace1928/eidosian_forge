from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_Exec(self, node):
    self._new_line()
    self._write('exec ')
    self.visit(node.body)
    if not node.globals:
        return
    self._write(', ')
    self.visit(node.globals)
    if not node.locals:
        return
    self._write(', ')
    self.visit(node.locals)