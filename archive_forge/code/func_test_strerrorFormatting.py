import os
import socket
from unittest import skipIf
from twisted.internet.tcp import ECONNABORTED
from twisted.python.runtime import platform
from twisted.python.win32 import _ErrorFormatter, formatError
from twisted.trial.unittest import TestCase
def test_strerrorFormatting(self):
    """
        L{_ErrorFormatter.formatError} should use L{os.strerror} to format
        error messages if it is constructed without any better mechanism.
        """
    formatter = _ErrorFormatter(None, None, None)
    message = formatter.formatError(self.probeErrorCode)
    self.assertEqual(message, os.strerror(self.probeErrorCode))