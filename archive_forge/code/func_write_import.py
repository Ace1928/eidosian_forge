import ast
from .qt import ClassFlag, qt_class_flags
def write_import(file, i_node):
    """Print an import of a Qt class as #include"""
    for alias in i_node.names:
        if alias.name.startswith('Q'):
            file.write(f'#include <{alias.name}>\n')