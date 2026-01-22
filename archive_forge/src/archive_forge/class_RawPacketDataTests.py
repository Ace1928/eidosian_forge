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
@skipIf(not cryptography, 'Cannot run without cryptography')
class RawPacketDataTests(TestCase):
    """
    Tests for L{filetransfer.FileTransferClient} which explicitly craft certain
    less common protocol messages to exercise their handling.
    """

    def setUp(self):
        self.ftc = filetransfer.FileTransferClient()

    def test_packetSTATUS(self):
        """
        A STATUS packet containing a result code, a message, and a language is
        parsed to produce the result of an outstanding request L{Deferred}.

        @see: U{section 9.1<http://tools.ietf.org/html/draft-ietf-secsh-filexfer-13#section-9.1>}
            of the SFTP Internet-Draft.
        """
        d = defer.Deferred()
        d.addCallback(self._cbTestPacketSTATUS)
        self.ftc.openRequests[1] = d
        data = struct.pack('!LL', 1, filetransfer.FX_OK) + common.NS(b'msg') + common.NS(b'lang')
        self.ftc.packet_STATUS(data)
        return d

    def _cbTestPacketSTATUS(self, result):
        """
        Assert that the result is a two-tuple containing the message and
        language from the STATUS packet.
        """
        self.assertEqual(result[0], b'msg')
        self.assertEqual(result[1], b'lang')

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

    def _cbTestPacketSTATUSShort(self, result):
        """
        Assert that the result is a two-tuple containing empty strings, since
        the STATUS packet had neither a message nor a language.
        """
        self.assertEqual(result[0], b'')
        self.assertEqual(result[1], b'')

    def test_packetSTATUSWithoutLang(self):
        """
        A STATUS packet containing a result code and a message but no language
        can also be parsed to produce the result of an outstanding request
        L{Deferred}.  Such packets are sent by some SFTP implementations, though
        not strictly legal.

        @see: U{section 9.1<http://tools.ietf.org/html/draft-ietf-secsh-filexfer-13#section-9.1>}
            of the SFTP Internet-Draft.
        """
        d = defer.Deferred()
        d.addCallback(self._cbTestPacketSTATUSWithoutLang)
        self.ftc.openRequests[1] = d
        data = struct.pack('!LL', 1, filetransfer.FX_OK) + common.NS(b'msg')
        self.ftc.packet_STATUS(data)
        return d

    def _cbTestPacketSTATUSWithoutLang(self, result):
        """
        Assert that the result is a two-tuple containing the message from the
        STATUS packet and an empty string, since the language was missing.
        """
        self.assertEqual(result[0], b'msg')
        self.assertEqual(result[1], b'')