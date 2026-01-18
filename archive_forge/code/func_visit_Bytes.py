from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_Bytes(self, node):
    self._write(repr(node.s))