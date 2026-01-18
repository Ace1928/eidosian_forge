from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_GeneratorExp(self, node):
    self._write('(')
    self.visit(node.elt)
    for generator in node.generators:
        self._write(' for ')
        self.visit(generator.target)
        self._write(' in ')
        self.visit(generator.iter)
        for ifexpr in generator.ifs:
            self._write(' if ')
            self.visit(ifexpr)
    self._write(')')