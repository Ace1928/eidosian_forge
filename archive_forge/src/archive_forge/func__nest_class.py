import ast
import sys
import importlib.util
def _nest_class(ob, class_name, lineno, end_lineno, super=None):
    """Return a Class after nesting within ob."""
    return Class(ob.module, class_name, super, ob.file, lineno, parent=ob, end_lineno=end_lineno)