from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_With(self, node):
    self._new_line()
    self._write('with ')
    items = getattr(node, 'items', None)
    first = True
    if items is None:
        items = [node]
    for item in items:
        if not first:
            self._write(', ')
        first = False
        self.visit(item.context_expr)
        if getattr(item, 'optional_vars', None):
            self._write(' as ')
            self.visit(item.optional_vars)
    self._write(':')
    self._change_indent(1)
    for statement in node.body:
        self.visit(statement)
    self._change_indent(-1)