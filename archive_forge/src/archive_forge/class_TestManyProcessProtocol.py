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
class TestManyProcessProtocol(TestProcessProtocol):

    def __init__(self):
        self.deferred = defer.Deferred()

    def processEnded(self, reason):
        self.reason = reason
        if reason.check(error.ProcessDone):
            self.deferred.callback(None)
        else:
            self.deferred.errback(reason)