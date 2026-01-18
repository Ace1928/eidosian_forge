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
def test_normalTermination(self):
    cmd = self.getCommand('true')
    d = defer.Deferred()
    p = TrivialProcessProtocol(d)
    reactor.spawnProcess(p, cmd, [b'true'], env=None, usePTY=self.usePTY)

    def check(ignored):
        p.reason.trap(error.ProcessDone)
        self.assertEqual(p.reason.value.exitCode, 0)
        self.assertIsNone(p.reason.value.signal)
    d.addCallback(check)
    return d