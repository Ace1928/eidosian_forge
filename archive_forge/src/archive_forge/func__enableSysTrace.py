import sys, re, traceback, threading
from .. import exceptionHandling as exceptionHandling
from ..Qt import QtWidgets, QtCore
from ..functions import SignalBlock
from .stackwidget import StackWidget
def _enableSysTrace(self):
    sys.settrace(self.systrace)
    threading.settrace(self.systrace)
    if hasattr(threading, 'settrace_all_threads'):
        threading.settrace_all_threads(self.systrace)