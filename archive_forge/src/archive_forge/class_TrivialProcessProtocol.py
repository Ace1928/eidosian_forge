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
class TrivialProcessProtocol(protocol.ProcessProtocol):
    """
    Simple process protocol for tests purpose.

    @ivar outData: data received from stdin
    @ivar errData: data received from stderr
    """

    def __init__(self, d):
        """
        Create the deferred that will be fired at the end, and initialize
        data structures.
        """
        self.deferred = d
        self.outData = []
        self.errData = []

    def processEnded(self, reason):
        self.reason = reason
        self.deferred.callback(None)

    def outReceived(self, data):
        self.outData.append(data)

    def errReceived(self, data):
        self.errData.append(data)