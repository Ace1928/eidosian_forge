import sys, re, traceback, threading
from .. import exceptionHandling as exceptionHandling
from ..Qt import QtWidgets, QtCore
from ..functions import SignalBlock
from .stackwidget import StackWidget
def updateSysTrace(self):
    if not self.catchNextExceptionBtn.isChecked() and (not self.catchAllExceptionsBtn.isChecked()):
        if sys.gettrace() == self.systrace:
            self._disableSysTrace()
        return
    if self.onlyUncaughtCheck.isChecked():
        if sys.gettrace() == self.systrace:
            self._disableSysTrace()
    elif sys.gettrace() not in (None, self.systrace):
        self.onlyUncaughtCheck.setChecked(False)
        raise Exception('sys.settrace is in use (are you using another debugger?); cannot monitor for caught exceptions.')
    else:
        self._enableSysTrace()