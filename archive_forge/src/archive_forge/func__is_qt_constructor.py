import ast
import sys
import tokenize
import warnings
from .formatter import (CppFormatter, format_for_loop, format_literal,
from .nodedump import debug_format_node
from .qt import ClassFlag, qt_class_flags
def _is_qt_constructor(assign_node):
    """Is this assignment node a plain construction of a Qt class?
       'f = QFile(name)'. Returns the class_name."""
    call = assign_node.value
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
        func = call.func.id
        if func.startswith('Q'):
            return func
    return None