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
class TwoProcessProtocol(protocol.ProcessProtocol):
    num = -1
    finished = 0

    def __init__(self):
        self.deferred = defer.Deferred()

    def outReceived(self, data):
        pass

    def processEnded(self, reason):
        self.finished = 1
        self.deferred.callback(None)