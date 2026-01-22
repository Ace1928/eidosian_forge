import logging
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.qtproxies import (QtWidgets, QtGui, Literal,
class _ModuleWrapper(object):

    def __init__(self, name, classes):
        if '.' in name:
            idx = name.rfind('.')
            self._package = name[:idx]
            self._module = name[idx + 1:]
        else:
            self._package = None
            self._module = name
        self._classes = set(classes)
        self._used = False

    def search(self, cls):
        if cls in self._classes:
            self._used = True
            return type(cls, (QtWidgets.QWidget,), {'module': self._module})
        else:
            return None

    def _writeImportCode(self):
        if self._used:
            if self._package is None:
                write_code('import %s' % self._module)
            else:
                write_code('from %s import %s' % (self._package, self._module))