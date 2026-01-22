import logging
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.qtproxies import (QtWidgets, QtGui, Literal,
class _QtGuiWrapper(object):

    def search(clsname):
        try:
            return strict_getattr(QtGui, clsname)
        except AttributeError:
            return None
    search = staticmethod(search)