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
class FileTransferForTestAvatar(SFTPServerForUnixConchUser):

    def gotVersion(self, version, otherExt):
        return {b'conchTest': b'ext data'}

    def extendedRequest(self, extName, extData):
        if extName == b'testExtendedRequest':
            return b'bar'
        raise NotImplementedError