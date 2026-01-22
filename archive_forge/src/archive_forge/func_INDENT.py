import ast
from .qt import ClassFlag, qt_class_flags
def INDENT(self):
    """Write indentation"""
    self._output_file.write(self._indentation)