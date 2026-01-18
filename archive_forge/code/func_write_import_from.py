import ast
from .qt import ClassFlag, qt_class_flags
def write_import_from(file, i_node):
    """Print an import from Qt classes as #include sequence"""
    mod = i_node.module
    if not mod.startswith('PySide') and (not mod.startswith('PyQt')):
        return
    dot = mod.find('.')
    qt_module = mod[dot + 1:] + '/' if dot >= 0 else ''
    for i in i_node.names:
        if i.name.startswith('Q'):
            file.write(f'#include <{qt_module}{i.name}>\n')