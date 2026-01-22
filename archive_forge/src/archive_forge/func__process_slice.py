from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def _process_slice(node):
    if isinstance(node, _ast_Ellipsis):
        self._write('...')
    elif isinstance(node, _ast.Slice):
        if getattr(node, 'lower', 'None'):
            self.visit(node.lower)
        self._write(':')
        if getattr(node, 'upper', None):
            self.visit(node.upper)
        if getattr(node, 'step', None):
            self._write(':')
            self.visit(node.step)
    elif isinstance(node, _ast.Index):
        self.visit(node.value)
    elif isinstance(node, _ast.ExtSlice):
        self.visit(node.dims[0])
        for dim in node.dims[1:]:
            self._write(', ')
            self.visit(dim)
    else:
        self.visit(node)