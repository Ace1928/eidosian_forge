from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def _new_line(self):
    if self.line is not None:
        self.code += self.line + '\n'
        self.lines_info.append(self.line_info)
    self.line = ' ' * 4 * self.indent
    if len(self.blame_stack) == 0:
        self.line_info = []
        self.last = None
    else:
        self.line_info = [(0, self.blame_stack[-1])]
        self.last = self.blame_stack[-1]