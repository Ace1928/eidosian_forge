import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
class Win32SignalProtocol(SignalProtocol):
    """
    A win32-specific process protocol that handles C{processEnded}
    differently: processes should exit with exit code 1.
    """

    def processEnded(self, reason):
        """
        Callback C{self.deferred} with L{None} if C{reason} is a
        L{error.ProcessTerminated} failure with C{exitCode} set to 1.
        Otherwise, errback with a C{ValueError} describing the problem.
        """
        if not reason.check(error.ProcessTerminated):
            return self.deferred.errback(ValueError(f'wrong termination: {reason}'))
        v = reason.value
        if v.exitCode != 1:
            return self.deferred.errback(ValueError(f'Wrong exit code: {v.exitCode}'))
        self.deferred.callback(None)