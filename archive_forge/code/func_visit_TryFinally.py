from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_TryFinally(self, node):
    self._new_line()
    self._write('try:')
    self._change_indent(1)
    for statement in node.body:
        self.visit(statement)
    self._change_indent(-1)
    if getattr(node, 'finalbody', None):
        self._new_line()
        self._write('finally:')
        self._change_indent(1)
        for statement in node.finalbody:
            self.visit(statement)
        self._change_indent(-1)