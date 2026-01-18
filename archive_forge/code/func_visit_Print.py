from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_Print(self, node):
    self._new_line()
    self._write('print')
    if getattr(node, 'dest', None):
        self._write(' >> ')
        self.visit(node.dest)
        if getattr(node, 'values', None):
            self._write(', ')
    else:
        self._write(' ')
    if getattr(node, 'values', None):
        self.visit(node.values[0])
        for value in node.values[1:]:
            self._write(', ')
            self.visit(value)
    if not node.nl:
        self._write(',')