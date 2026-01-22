import logging
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.qtproxies import (QtWidgets, QtGui, Literal,
def _resolveBaseclass(self, baseClass):
    try:
        for x in range(0, 10):
            try:
                return strict_getattr(QtWidgets, baseClass)
            except AttributeError:
                pass
            baseClass = self._widgets[baseClass][0]
        else:
            raise ValueError('baseclass resolve took too long, check custom widgets')
    except KeyError:
        raise ValueError('unknown baseclass %s' % baseClass)