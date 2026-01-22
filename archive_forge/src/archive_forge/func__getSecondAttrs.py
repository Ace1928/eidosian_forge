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
def _getSecondAttrs(firstAttrs):
    d = self.client.getAttrs(b'testfile1')
    self._emptyBuffers()
    d.addCallback(self.assertEqual, firstAttrs)
    return d