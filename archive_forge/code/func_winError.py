import os
import socket
from unittest import skipIf
from twisted.internet.tcp import ECONNABORTED
from twisted.python.runtime import platform
from twisted.python.win32 import _ErrorFormatter, formatError
from twisted.trial.unittest import TestCase
def winError(errorCode):
    winCalls.append(errorCode)
    return _MyWindowsException(errorCode, self.probeMessage)