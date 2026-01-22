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
def _readChunk2(_, openFile):
    d = openFile.readChunk(0, 30)
    self._emptyBuffers()
    d.addCallback(self.assertEqual, b'a' * 10 + b'b' * 10 + b'c' * 10)
    return d