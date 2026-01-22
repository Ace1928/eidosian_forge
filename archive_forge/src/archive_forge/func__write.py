from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def _write(self, s):
    if len(s) == 0:
        return
    if len(self.blame_stack) == 0:
        if self.last is not None:
            self.last = None
            self.line_info.append((len(self.line), self.last))
    elif self.last != self.blame_stack[-1]:
        self.last = self.blame_stack[-1]
        self.line_info.append((len(self.line), self.last))
    self.line += s