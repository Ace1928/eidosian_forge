from .error import MarkedYAMLError
from .tokens import *
def unwind_indent(self, column):
    if self.flow_level:
        return
    while self.indent > column:
        mark = self.get_mark()
        self.indent = self.indents.pop()
        self.tokens.append(BlockEndToken(mark, mark))