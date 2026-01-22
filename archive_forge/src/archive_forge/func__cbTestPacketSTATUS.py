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
def _cbTestPacketSTATUS(self, result):
    """
        Assert that the result is a two-tuple containing the message and
        language from the STATUS packet.
        """
    self.assertEqual(result[0], b'msg')
    self.assertEqual(result[1], b'lang')