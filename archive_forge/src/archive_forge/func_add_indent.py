from .error import MarkedYAMLError
from .tokens import *
def add_indent(self, column):
    if self.indent < column:
        self.indents.append(self.indent)
        self.indent = column
        return True
    return False