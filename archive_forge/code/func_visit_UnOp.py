import ast
import sys
import tokenize
import warnings
from .formatter import (CppFormatter, format_for_loop, format_literal,
from .nodedump import debug_format_node
from .qt import ClassFlag, qt_class_flags
def visit_UnOp(self, node):
    self.generic_visit(node)