import ast
import sys
import tokenize
import warnings
from .formatter import (CppFormatter, format_for_loop, format_literal,
from .nodedump import debug_format_node
from .qt import ClassFlag, qt_class_flags
def _format_call(self, node):
    if self._ignore_function_def_node(node):
        return
    f = node.func
    if isinstance(f, ast.Name):
        self._output_file.write(f.id)
    else:
        names = []
        n = f
        while isinstance(n, ast.Attribute):
            names.insert(0, n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            if n.id != 'self':
                sep = '->'
                if n.id in self._stack_variables:
                    sep = '.'
                elif n.id[0:1].isupper():
                    sep = '::'
                self._output_file.write(n.id)
                self._output_file.write(sep)
        elif isinstance(n, ast.Call):
            self._format_call(n)
            self._output_file.write('->')
        self._output_file.write('->'.join(names))
    self._output_file.write('(')
    self._write_function_args(node.args)
    self._output_file.write(')')