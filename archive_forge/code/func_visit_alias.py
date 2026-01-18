from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_alias(self, node):
    self._write(node.name)
    if getattr(node, 'asname', None):
        self._write(' as ')
        self._write(node.asname)