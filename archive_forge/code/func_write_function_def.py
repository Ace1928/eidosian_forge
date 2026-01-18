import ast
from .qt import ClassFlag, qt_class_flags
def write_function_def(self, f_node, class_context):
    """Print a function definition with arguments"""
    self._output_file.write('\n')
    arguments = format_function_def_arguments(f_node)
    if f_node.name == '__init__' and class_context:
        name = class_context
    elif f_node.name == '__del__' and class_context:
        name = '~' + class_context
    else:
        return_type = 'void'
        if f_node.returns and isinstance(f_node.returns, ast.Name):
            return_type = _fix_function_argument_type(f_node.returns.id, True)
        name = return_type + ' ' + f_node.name
    self.indent_string(f'{name}({arguments})')
    self._output_file.write('\n')
    self.indent_line('{')