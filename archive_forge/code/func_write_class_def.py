import ast
from .qt import ClassFlag, qt_class_flags
def write_class_def(self, class_node):
    """Print a class definition with inheritance"""
    self._output_file.write('\n')
    inherits = format_inheritance(class_node)
    self.indent_line(f'class {class_node.name}{inherits}')
    self.indent_line('{')
    self.indent_line('public:')