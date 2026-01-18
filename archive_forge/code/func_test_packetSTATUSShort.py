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
def test_packetSTATUSShort(self):
    """
        A STATUS packet containing only a result code can also be parsed to
        produce the result of an outstanding request L{Deferred}.  Such packets
        are sent by some SFTP implementations, though not strictly legal.

        @see: U{section 9.1<http://tools.ietf.org/html/draft-ietf-secsh-filexfer-13#section-9.1>}
            of the SFTP Internet-Draft.
        """
    d = defer.Deferred()
    d.addCallback(self._cbTestPacketSTATUSShort)
    self.ftc.openRequests[1] = d
    data = struct.pack('!LL', 1, filetransfer.FX_OK)
    self.ftc.packet_STATUS(data)
    return d