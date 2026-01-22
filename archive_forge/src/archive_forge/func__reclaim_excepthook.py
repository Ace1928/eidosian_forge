import sys
import os
from IPython.external.qt_for_kernel import QtCore, QtGui, enum_helper
from IPython import get_ipython
def _reclaim_excepthook():
    shell = get_ipython()
    if shell is not None:
        sys.excepthook = shell.excepthook