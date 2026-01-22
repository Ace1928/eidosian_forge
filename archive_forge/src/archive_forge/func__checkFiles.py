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
def _checkFiles(ignored):
    fs = list(list(zip(*files))[0])
    fs.sort()
    self.assertEqual(fs, [b'.testHiddenFile', b'testDirectory', b'testRemoveFile', b'testRenameFile', b'testfile1'])