from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_Delete(self, node):
    self._new_line()
    self._write('del ')
    self.visit(node.targets[0])
    for target in node.targets[1:]:
        self._write(', ')
        self.visit(target)