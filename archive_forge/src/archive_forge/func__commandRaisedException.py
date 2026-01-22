import os
import sys
import pickle
import subprocess
from .. import getConfigOption
from ..Qt import QtCore, QtWidgets
from .repl_widget import ReplWidget
from .exception_widget import ExceptionHandlerWidget
def _commandRaisedException(self, repl, exc):
    self.excHandler.exceptionHandler(exc)