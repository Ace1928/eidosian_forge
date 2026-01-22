import logging
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.qtproxies import (QtWidgets, QtGui, Literal,
class _QtWidgetsWrapper(object):

    def search(clsname):
        try:
            return strict_getattr(QtWidgets, clsname)
        except AttributeError:
            return None
    search = staticmethod(search)