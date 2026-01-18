from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_Continue(self, node):
    self._new_line()
    self._write('continue')