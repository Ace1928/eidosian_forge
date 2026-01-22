import ast
import sys
import tokenize
import warnings
from .formatter import (CppFormatter, format_for_loop, format_literal,
from .nodedump import debug_format_node
from .qt import ClassFlag, qt_class_flags
def _debug_enter(self, node, parent=None):
    message = '{}>generic_visit({})'.format('  ' * self._debug_indent, debug_format_node(node))
    if parent:
        message += ', parent={}'.format(debug_format_node(parent))
    message += '\n'
    sys.stderr.write(message)
    self._debug_indent += 1