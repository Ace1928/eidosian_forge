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
def test_pid(self):
    """
        Simple test for the pid attribute of Process on win32.
        Launch process with mock win32process. The only mock aspect of this
        module is that the pid of the process created will always be 42.
        """
    from twisted.internet import _dumbwin32proc
    from twisted.test import mock_win32process
    self.patch(_dumbwin32proc, 'win32process', mock_win32process)
    scriptPath = FilePath(__file__).sibling('process_cmdline.py').path
    pyExe = FilePath(sys.executable).path
    d = defer.Deferred()
    processProto = TrivialProcessProtocol(d)
    comspec = 'cmd.exe'
    cmd = [comspec, '/c', pyExe, scriptPath]
    p = _dumbwin32proc.Process(reactor, processProto, None, cmd, {}, None)
    self.assertEqual(42, p.pid)
    self.assertEqual('<Process pid=42>', repr(p))

    def pidCompleteCb(result):
        self.assertIsNone(p.pid)
    return d.addCallback(pidCompleteCb)