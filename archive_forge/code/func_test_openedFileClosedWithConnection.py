import os
import re
import struct
from unittest import skipIf
from hamcrest import assert_that, equal_to
from twisted.internet import defer
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import StringTransport
from twisted.protocols import loopback
from twisted.python import components
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
def test_openedFileClosedWithConnection(self):
    """
        A file opened with C{openFile} is closed when the connection is lost.
        """
    d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
    self._emptyBuffers()
    oldClose = os.close
    closed = []

    def close(fd):
        closed.append(fd)
        oldClose(fd)
    self.patch(os, 'close', close)

    def _fileOpened(openFile):
        fd = self.server.openFiles[openFile.handle[4:]].fd
        self.serverTransport.loseConnection()
        self.clientTransport.loseConnection()
        self.serverTransport.clearBuffer()
        self.clientTransport.clearBuffer()
        self.assertEqual(self.server.openFiles, {})
        self.assertIn(fd, closed)
    d.addCallback(_fileOpened)
    return d