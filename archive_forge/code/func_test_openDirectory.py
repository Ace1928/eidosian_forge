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
def test_openDirectory(self):
    d = self.client.openDirectory(b'')
    self._emptyBuffers()
    files = []

    def _getFiles(openDir):

        def append(f):
            files.append(f)
            return openDir
        d = defer.maybeDeferred(openDir.next)
        self._emptyBuffers()
        d.addCallback(append)
        d.addCallback(_getFiles)
        d.addErrback(_close, openDir)
        return d

    def _checkFiles(ignored):
        fs = list(list(zip(*files))[0])
        fs.sort()
        self.assertEqual(fs, [b'.testHiddenFile', b'testDirectory', b'testRemoveFile', b'testRenameFile', b'testfile1'])

    def _close(_, openDir):
        d = openDir.close()
        self._emptyBuffers()
        return d
    d.addCallback(_getFiles)
    d.addCallback(_checkFiles)
    return d