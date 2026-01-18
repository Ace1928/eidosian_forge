from time import perf_counter
from ..Qt import QtCore, QtGui, QtWidgets
def wasCanceled(self):
    if self.disabled:
        return False
    return QtWidgets.QProgressDialog.wasCanceled(self)