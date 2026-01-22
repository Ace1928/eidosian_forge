import binascii
import re
import string
import struct
import types
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Dict, List, Optional, Tuple, Type
from twisted import __version__ as twisted_version
from twisted.conch.error import ConchError
from twisted.conch.ssh import _kex, address, service
from twisted.internet import defer
from twisted.protocols import loopback
from twisted.python import randbytes
from twisted.python.compat import iterbytes
from twisted.python.randbytes import insecureRandom
from twisted.python.reflect import requireModule
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class BaseSSHTransportTests(BaseSSHTransportBaseCase, TransportTestCase):
    """
    Test TransportBase. It implements the non-server/client specific
    parts of the SSH transport protocol.
    """
    if dependencySkip:
        skip = dependencySkip
    _A_KEXINIT_MESSAGE = b'\xaa' * 16 + common.NS(b'diffie-hellman-group14-sha1') + common.NS(b'ssh-rsa') + common.NS(b'aes256-ctr') + common.NS(b'aes256-ctr') + common.NS(b'hmac-sha1') + common.NS(b'hmac-sha1') + common.NS(b'none') + common.NS(b'none') + common.NS(b'') + common.NS(b'') + b'\x00' + b'\x00\x00\x00\x00'

    def test_sendVersion(self):
        """
        Test that the first thing sent over the connection is the version
        string.  The 'softwareversion' part must consist of printable
        US-ASCII characters, with the exception of whitespace characters and
        the minus sign.

        RFC 4253, section 4.2.
        """
        version = self.transport.value().split(b'\r\n', 1)[0]
        self.assertEqual(version, b'SSH-2.0-Twisted_' + twisted_version.encode('ascii'))
        softwareVersion = version.decode('ascii')[len('SSH-2.0-'):]
        softwareVersionRegex = '^(' + '|'.join((re.escape(c) for c in string.printable if c != '-' and (not c.isspace()))) + ')*$'
        self.assertRegex(softwareVersion, softwareVersionRegex)

    def test_dataReceiveVersionNotSentMemoryDOS(self):
        """
        When the peer is not sending its SSH version but keeps sending data,
        the connection is disconnected after 4KB to prevent buffering too
        much and running our of memory.
        """
        sut = MockTransportBase()
        sut.makeConnection(self.transport)
        sut.dataReceived(b'SSH-2-Server-Identifier')
        sut.dataReceived(b'1234567890' * 406)
        sut.dataReceived(b'1235678')
        self.assertFalse(self.transport.disconnecting)
        sut.dataReceived(b'1234567')
        self.assertTrue(self.transport.disconnecting)
        self.assertIn(b'Preventing a denial of service attack', self.transport.value())

    def test_sendPacketPlain(self):
        """
        Test that plain (unencrypted, uncompressed) packets are sent
        correctly.  The format is::
            uint32 length (including type and padding length)
            byte padding length
            byte type
            bytes[length-padding length-2] data
            bytes[padding length] padding
        """
        proto = MockTransportBase()
        proto.makeConnection(self.transport)
        self.finishKeyExchange(proto)
        self.transport.clear()
        message = ord('A')
        payload = b'BCDEFG'
        proto.sendPacket(message, payload)
        value = self.transport.value()
        self.assertEqual(value, b'\x00\x00\x00\x0c\x04ABCDEFG\x99\x99\x99\x99')

    def test_sendPacketEncrypted(self):
        """
        Test that packets sent while encryption is enabled are sent
        correctly.  The whole packet should be encrypted.
        """
        proto = MockTransportBase()
        proto.makeConnection(self.transport)
        self.finishKeyExchange(proto)
        proto.currentEncryptions = testCipher = MockCipher()
        message = ord('A')
        payload = b'BC'
        self.transport.clear()
        proto.sendPacket(message, payload)
        self.assertTrue(testCipher.usedEncrypt)
        value = self.transport.value()
        self.assertEqual(value, b'\x00\x00\x00\x08\x04ABC\x99\x99\x99\x99\x02')

    def test_sendPacketCompressed(self):
        """
        Test that packets sent while compression is enabled are sent
        correctly.  The packet type and data should be encrypted.
        """
        proto = MockTransportBase()
        proto.makeConnection(self.transport)
        self.finishKeyExchange(proto)
        proto.outgoingCompression = MockCompression()
        self.transport.clear()
        proto.sendPacket(ord('A'), b'B')
        value = self.transport.value()
        self.assertEqual(value, b'\x00\x00\x00\x0c\x08BAf\x99\x99\x99\x99\x99\x99\x99\x99')

    def test_sendPacketBoth(self):
        """
        Test that packets sent while compression and encryption are
        enabled are sent correctly.  The packet type and data should be
        compressed and then the whole packet should be encrypted.
        """
        proto = MockTransportBase()
        proto.makeConnection(self.transport)
        self.finishKeyExchange(proto)
        proto.currentEncryptions = testCipher = MockCipher()
        proto.outgoingCompression = MockCompression()
        message = ord('A')
        payload = b'BC'
        self.transport.clear()
        proto.sendPacket(message, payload)
        self.assertTrue(testCipher.usedEncrypt)
        value = self.transport.value()
        self.assertEqual(value, b'\x00\x00\x00\x0e\tCBAf\x99\x99\x99\x99\x99\x99\x99\x99\x99\x02')

    def test_getPacketPlain(self):
        """
        Test that packets are retrieved correctly out of the buffer when
        no encryption is enabled.
        """
        proto = MockTransportBase()
        proto.makeConnection(self.transport)
        self.finishKeyExchange(proto)
        self.transport.clear()
        proto.sendPacket(ord('A'), b'BC')
        proto.buf = self.transport.value() + b'extra'
        self.assertEqual(proto.getPacket(), b'ABC')
        self.assertEqual(proto.buf, b'extra')

    def test_getPacketEncrypted(self):
        """
        Test that encrypted packets are retrieved correctly.
        See test_sendPacketEncrypted.
        """
        proto = MockTransportBase()
        proto.sendKexInit = lambda: None
        proto.makeConnection(self.transport)
        self.transport.clear()
        proto.currentEncryptions = testCipher = MockCipher()
        proto.sendPacket(ord('A'), b'BCD')
        value = self.transport.value()
        proto.buf = value[:MockCipher.decBlockSize]
        self.assertIsNone(proto.getPacket())
        self.assertTrue(testCipher.usedDecrypt)
        self.assertEqual(proto.first, b'\x00\x00\x00\x0e\tA')
        proto.buf += value[MockCipher.decBlockSize:]
        self.assertEqual(proto.getPacket(), b'ABCD')
        self.assertEqual(proto.buf, b'')

    def test_getPacketCompressed(self):
        """
        Test that compressed packets are retrieved correctly.  See
        test_sendPacketCompressed.
        """
        proto = MockTransportBase()
        proto.makeConnection(self.transport)
        self.finishKeyExchange(proto)
        self.transport.clear()
        proto.outgoingCompression = MockCompression()
        proto.incomingCompression = proto.outgoingCompression
        proto.sendPacket(ord('A'), b'BCD')
        proto.buf = self.transport.value()
        self.assertEqual(proto.getPacket(), b'ABCD')

    def test_getPacketBoth(self):
        """
        Test that compressed and encrypted packets are retrieved correctly.
        See test_sendPacketBoth.
        """
        proto = MockTransportBase()
        proto.sendKexInit = lambda: None
        proto.makeConnection(self.transport)
        self.transport.clear()
        proto.currentEncryptions = MockCipher()
        proto.outgoingCompression = MockCompression()
        proto.incomingCompression = proto.outgoingCompression
        proto.sendPacket(ord('A'), b'BCDEFG')
        proto.buf = self.transport.value()
        self.assertEqual(proto.getPacket(), b'ABCDEFG')

    def test_ciphersAreValid(self):
        """
        Test that all the supportedCiphers are valid.
        """
        ciphers = transport.SSHCiphers(b'A', b'B', b'C', b'D')
        iv = key = b'\x00' * 16
        for cipName in self.proto.supportedCiphers:
            self.assertTrue(ciphers._getCipher(cipName, iv, key))

    def test_sendKexInit(self):
        """
        Test that the KEXINIT (key exchange initiation) message is sent
        correctly.  Payload::
            bytes[16] cookie
            string key exchange algorithms
            string public key algorithms
            string outgoing ciphers
            string incoming ciphers
            string outgoing MACs
            string incoming MACs
            string outgoing compressions
            string incoming compressions
            bool first packet follows
            uint32 0
        """
        value = self.transport.value().split(b'\r\n', 1)[1]
        self.proto.buf = value
        packet = self.proto.getPacket()
        self.assertEqual(packet[0:1], bytes((transport.MSG_KEXINIT,)))
        self.assertEqual(packet[1:17], b'\x99' * 16)
        keyExchanges, pubkeys, ciphers1, ciphers2, macs1, macs2, compressions1, compressions2, languages1, languages2, buf = common.getNS(packet[17:], 10)
        self.assertEqual(keyExchanges, b','.join(self.proto.supportedKeyExchanges + [b'ext-info-s']))
        self.assertEqual(pubkeys, b','.join(self.proto.supportedPublicKeys))
        self.assertEqual(ciphers1, b','.join(self.proto.supportedCiphers))
        self.assertEqual(ciphers2, b','.join(self.proto.supportedCiphers))
        self.assertEqual(macs1, b','.join(self.proto.supportedMACs))
        self.assertEqual(macs2, b','.join(self.proto.supportedMACs))
        self.assertEqual(compressions1, b','.join(self.proto.supportedCompressions))
        self.assertEqual(compressions2, b','.join(self.proto.supportedCompressions))
        self.assertEqual(languages1, b','.join(self.proto.supportedLanguages))
        self.assertEqual(languages2, b','.join(self.proto.supportedLanguages))
        self.assertEqual(buf, b'\x00' * 5)

    def test_receiveKEXINITReply(self):
        """
        Immediately after connecting, the transport expects a KEXINIT message
        and does not reply to it.
        """
        self.transport.clear()
        self.proto.dispatchMessage(transport.MSG_KEXINIT, self._A_KEXINIT_MESSAGE)
        self.assertEqual(self.packets, [])

    def test_sendKEXINITReply(self):
        """
        When a KEXINIT message is received which is not a reply to an earlier
        KEXINIT message which was sent, a KEXINIT reply is sent.
        """
        self.finishKeyExchange(self.proto)
        del self.packets[:]
        self.proto.dispatchMessage(transport.MSG_KEXINIT, self._A_KEXINIT_MESSAGE)
        self.assertEqual(len(self.packets), 1)
        self.assertEqual(self.packets[0][0], transport.MSG_KEXINIT)

    def test_sendKexInitTwiceFails(self):
        """
        A new key exchange cannot be started while a key exchange is already in
        progress.  If an attempt is made to send a I{KEXINIT} message using
        L{SSHTransportBase.sendKexInit} while a key exchange is in progress
        causes that method to raise a L{RuntimeError}.
        """
        self.assertRaises(RuntimeError, self.proto.sendKexInit)

    def test_sendKexInitBlocksOthers(self):
        """
        After L{SSHTransportBase.sendKexInit} has been called, messages types
        other than the following are queued and not sent until after I{NEWKEYS}
        is sent by L{SSHTransportBase._keySetup}.

        RFC 4253, section 7.1.
        """
        disallowedMessageTypes = [transport.MSG_SERVICE_REQUEST, transport.MSG_KEXINIT]
        self.transport.clear()
        del self.proto.sendPacket
        for messageType in disallowedMessageTypes:
            self.proto.sendPacket(messageType, b'foo')
            self.assertEqual(self.transport.value(), b'')
        self.finishKeyExchange(self.proto)
        self.proto.nextEncryptions = MockCipher()
        self.proto._newKeys()
        self.assertEqual(self.transport.value().count(b'foo'), 2)

    def test_sendExtInfo(self):
        """
        Test that EXT_INFO messages are sent correctly.  See RFC 8308,
        section 2.3.
        """
        self.proto._peerSupportsExtensions = True
        self.proto.sendExtInfo([(b'server-sig-algs', b'ssh-rsa,rsa-sha2-256'), (b'elevation', b'd')])
        self.assertEqual(self.packets, [(transport.MSG_EXT_INFO, b'\x00\x00\x00\x02' + common.NS(b'server-sig-algs') + common.NS(b'ssh-rsa,rsa-sha2-256') + common.NS(b'elevation') + common.NS(b'd'))])

    def test_sendExtInfoUnsupported(self):
        """
        If the peer has not advertised support for extension negotiation, no
        EXT_INFO message is sent, since RFC 8308 only guarantees that the
        peer will be prepared to accept it if it has advertised support.
        """
        self.proto.sendExtInfo([(b'server-sig-algs', b'ssh-rsa,rsa-sha2-256')])
        self.assertEqual(self.packets, [])

    def test_EXT_INFO(self):
        """
        When an EXT_INFO message is received, the transport stores a mapping
        of the peer's advertised extensions.  See RFC 8308, section 2.3.
        """
        self.proto.dispatchMessage(transport.MSG_EXT_INFO, b'\x00\x00\x00\x02' + common.NS(b'server-sig-algs') + common.NS(b'ssh-rsa,rsa-sha2-256,rsa-sha2-512') + common.NS(b'no-flow-control') + common.NS(b's'))
        self.assertEqual(self.proto.peerExtensions, {b'server-sig-algs': b'ssh-rsa,rsa-sha2-256,rsa-sha2-512', b'no-flow-control': b's'})

    def test_sendDebug(self):
        """
        Test that debug messages are sent correctly.  Payload::
            bool always display
            string debug message
            string language
        """
        self.proto.sendDebug(b'test', True, b'en')
        self.assertEqual(self.packets, [(transport.MSG_DEBUG, b'\x01\x00\x00\x00\x04test\x00\x00\x00\x02en')])

    def test_receiveDebug(self):
        """
        Test that debug messages are received correctly.  See test_sendDebug.
        """
        self.proto.dispatchMessage(transport.MSG_DEBUG, b'\x01\x00\x00\x00\x04test\x00\x00\x00\x02en')
        self.proto.dispatchMessage(transport.MSG_DEBUG, b'\x00\x00\x00\x00\x06silent\x00\x00\x00\x02en')
        self.assertEqual(self.proto.debugs, [(True, b'test', b'en'), (False, b'silent', b'en')])

    def test_sendIgnore(self):
        """
        Test that ignored messages are sent correctly.  Payload::
            string ignored data
        """
        self.proto.sendIgnore(b'test')
        self.assertEqual(self.packets, [(transport.MSG_IGNORE, b'\x00\x00\x00\x04test')])

    def test_receiveIgnore(self):
        """
        Test that ignored messages are received correctly.  See
        test_sendIgnore.
        """
        self.proto.dispatchMessage(transport.MSG_IGNORE, b'test')
        self.assertEqual(self.proto.ignoreds, [b'test'])

    def test_sendUnimplemented(self):
        """
        Test that unimplemented messages are sent correctly.  Payload::
            uint32 sequence number
        """
        self.proto.sendUnimplemented()
        self.assertEqual(self.packets, [(transport.MSG_UNIMPLEMENTED, b'\x00\x00\x00\x00')])

    def test_receiveUnimplemented(self):
        """
        Test that unimplemented messages are received correctly.  See
        test_sendUnimplemented.
        """
        self.proto.dispatchMessage(transport.MSG_UNIMPLEMENTED, b'\x00\x00\x00\xff')
        self.assertEqual(self.proto.unimplementeds, [255])

    def test_sendDisconnect(self):
        """
        Test that disconnection messages are sent correctly.  Payload::
            uint32 reason code
            string reason description
            string language
        """
        disconnected = [False]

        def stubLoseConnection():
            disconnected[0] = True
        self.transport.loseConnection = stubLoseConnection
        self.proto.sendDisconnect(255, b'test')
        self.assertEqual(self.packets, [(transport.MSG_DISCONNECT, b'\x00\x00\x00\xff\x00\x00\x00\x04test\x00\x00\x00\x00')])
        self.assertTrue(disconnected[0])

    def test_receiveDisconnect(self):
        """
        Test that disconnection messages are received correctly.  See
        test_sendDisconnect.
        """
        disconnected = [False]

        def stubLoseConnection():
            disconnected[0] = True
        self.transport.loseConnection = stubLoseConnection
        self.proto.dispatchMessage(transport.MSG_DISCONNECT, b'\x00\x00\x00\xff\x00\x00\x00\x04test')
        self.assertEqual(self.proto.errors, [(255, b'test')])
        self.assertTrue(disconnected[0])

    def test_dataReceived(self):
        """
        Test that dataReceived parses packets and dispatches them to
        ssh_* methods.
        """
        kexInit = [False]

        def stubKEXINIT(packet):
            kexInit[0] = True
        self.proto.ssh_KEXINIT = stubKEXINIT
        self.proto.dataReceived(self.transport.value())
        self.assertTrue(self.proto.gotVersion)
        self.assertEqual(self.proto.ourVersionString, self.proto.otherVersionString)
        self.assertTrue(kexInit[0])

    def test_service(self):
        """
        Test that the transport can set the running service and dispatches
        packets to the service's packetReceived method.
        """
        service = MockService()
        self.proto.setService(service)
        self.assertEqual(self.proto.service, service)
        self.assertTrue(service.started)
        self.proto.dispatchMessage(255, b'test')
        self.assertEqual(self.packets, [(255, b'test')])
        service2 = MockService()
        self.proto.setService(service2)
        self.assertTrue(service2.started)
        self.assertTrue(service.stopped)
        self.proto.connectionLost(None)
        self.assertTrue(service2.stopped)

    def test_avatar(self):
        """
        Test that the transport notifies the avatar of disconnections.
        """
        disconnected = [False]

        def logout():
            disconnected[0] = True
        self.proto.logoutFunction = logout
        self.proto.avatar = True
        self.proto.connectionLost(None)
        self.assertTrue(disconnected[0])

    def test_isEncrypted(self):
        """
        Test that the transport accurately reflects its encrypted status.
        """
        self.assertFalse(self.proto.isEncrypted('in'))
        self.assertFalse(self.proto.isEncrypted('out'))
        self.assertFalse(self.proto.isEncrypted('both'))
        self.proto.currentEncryptions = MockCipher()
        self.assertTrue(self.proto.isEncrypted('in'))
        self.assertTrue(self.proto.isEncrypted('out'))
        self.assertTrue(self.proto.isEncrypted('both'))
        self.proto.currentEncryptions = transport.SSHCiphers(b'none', b'none', b'none', b'none')
        self.assertFalse(self.proto.isEncrypted('in'))
        self.assertFalse(self.proto.isEncrypted('out'))
        self.assertFalse(self.proto.isEncrypted('both'))
        self.assertRaises(TypeError, self.proto.isEncrypted, 'bad')

    def test_isVerified(self):
        """
        Test that the transport accurately reflects its verified status.
        """
        self.assertFalse(self.proto.isVerified('in'))
        self.assertFalse(self.proto.isVerified('out'))
        self.assertFalse(self.proto.isVerified('both'))
        self.proto.currentEncryptions = MockCipher()
        self.assertTrue(self.proto.isVerified('in'))
        self.assertTrue(self.proto.isVerified('out'))
        self.assertTrue(self.proto.isVerified('both'))
        self.proto.currentEncryptions = transport.SSHCiphers(b'none', b'none', b'none', b'none')
        self.assertFalse(self.proto.isVerified('in'))
        self.assertFalse(self.proto.isVerified('out'))
        self.assertFalse(self.proto.isVerified('both'))
        self.assertRaises(TypeError, self.proto.isVerified, 'bad')

    def test_loseConnection(self):
        """
        Test that loseConnection sends a disconnect message and closes the
        connection.
        """
        disconnected = [False]

        def stubLoseConnection():
            disconnected[0] = True
        self.transport.loseConnection = stubLoseConnection
        self.proto.loseConnection()
        self.assertEqual(self.packets[0][0], transport.MSG_DISCONNECT)
        self.assertEqual(self.packets[0][1][3:4], bytes((transport.DISCONNECT_CONNECTION_LOST,)))

    def test_badVersion(self):
        """
        Test that the transport disconnects when it receives a bad version.
        """

        def testBad(version):
            self.packets = []
            self.proto.gotVersion = False
            disconnected = [False]

            def stubLoseConnection():
                disconnected[0] = True
            self.transport.loseConnection = stubLoseConnection
            for c in iterbytes(version + b'\r\n'):
                self.proto.dataReceived(c)
            self.assertTrue(disconnected[0])
            self.assertEqual(self.packets[0][0], transport.MSG_DISCONNECT)
            self.assertEqual(self.packets[0][1][3:4], bytes((transport.DISCONNECT_PROTOCOL_VERSION_NOT_SUPPORTED,)))
        testBad(b'SSH-1.5-OpenSSH')
        testBad(b'SSH-3.0-Twisted')
        testBad(b'GET / HTTP/1.1')

    def test_dataBeforeVersion(self):
        """
        Test that the transport ignores data sent before the version string.
        """
        proto = MockTransportBase()
        proto.makeConnection(proto_helpers.StringTransport())
        data = b"here's some stuff beforehand\nhere's some other stuff\n" + proto.ourVersionString + b'\r\n'
        [proto.dataReceived(c) for c in iterbytes(data)]
        self.assertTrue(proto.gotVersion)
        self.assertEqual(proto.otherVersionString, proto.ourVersionString)

    def test_compatabilityVersion(self):
        """
        Test that the transport treats the compatibility version (1.99)
        as equivalent to version 2.0.
        """
        proto = MockTransportBase()
        proto.makeConnection(proto_helpers.StringTransport())
        proto.dataReceived(b'SSH-1.99-OpenSSH\n')
        self.assertTrue(proto.gotVersion)
        self.assertEqual(proto.otherVersionString, b'SSH-1.99-OpenSSH')

    def test_dataReceivedSSHVersionUnixNewline(self):
        """
        It can parse the SSH version string even when it ends only in
        Unix newlines (CR) and does not follows the RFC 4253 to use
        network newlines (CR LF).
        """
        sut = MockTransportBase()
        sut.makeConnection(proto_helpers.StringTransport())
        sut.dataReceived(b'SSH-2.0-PoorSSHD Some-comment here\nmore-data')
        self.assertTrue(sut.gotVersion)
        self.assertEqual(sut.otherVersionString, b'SSH-2.0-PoorSSHD Some-comment here')

    def test_dataReceivedSSHVersionTrailingSpaces(self):
        """
        The trailing spaces from SSH version comment are not removed.

        The SSH version string needs to be kept as received
        (without CR LF end of line) as they are used in the host
        authentication process.

        This can happen with a Bitvise SSH server which hides its version.
        """
        sut = MockTransportBase()
        sut.makeConnection(proto_helpers.StringTransport())
        sut.dataReceived(b'SSH-2.0-9.99 FlowSsh: Bitvise SSH Server (WinSSHD) \r\nmore-data')
        self.assertTrue(sut.gotVersion)
        self.assertEqual(sut.otherVersionString, b'SSH-2.0-9.99 FlowSsh: Bitvise SSH Server (WinSSHD) ')

    def test_supportedVersionsAreAllowed(self):
        """
        If an unusual SSH version is received and is included in
        C{supportedVersions}, an unsupported version error is not emitted.
        """
        proto = MockTransportBase()
        proto.supportedVersions = (b'9.99',)
        proto.makeConnection(proto_helpers.StringTransport())
        proto.dataReceived(b'SSH-9.99-OpenSSH\n')
        self.assertFalse(proto.gotUnsupportedVersion)

    def test_unsupportedVersionsCallUnsupportedVersionReceived(self):
        """
        If an unusual SSH version is received and is not included in
        C{supportedVersions}, an unsupported version error is emitted.
        """
        proto = MockTransportBase()
        proto.supportedVersions = (b'2.0',)
        proto.makeConnection(proto_helpers.StringTransport())
        proto.dataReceived(b'SSH-9.99-OpenSSH\n')
        self.assertEqual(b'9.99', proto.gotUnsupportedVersion)

    def test_badPackets(self):
        """
        Test that the transport disconnects with an error when it receives
        bad packets.
        """

        def testBad(packet, error=transport.DISCONNECT_PROTOCOL_ERROR):
            self.packets = []
            self.proto.buf = packet
            self.assertIsNone(self.proto.getPacket())
            self.assertEqual(len(self.packets), 1)
            self.assertEqual(self.packets[0][0], transport.MSG_DISCONNECT)
            self.assertEqual(self.packets[0][1][3:4], bytes((error,)))
        testBad(b'\xff' * 8)
        testBad(b'\x00\x00\x00\x05\x00BCDE')
        oldEncryptions = self.proto.currentEncryptions
        self.proto.currentEncryptions = MockCipher()
        testBad(b'\x00\x00\x00\x08\x06AB123456', transport.DISCONNECT_MAC_ERROR)
        self.proto.currentEncryptions.decrypt = lambda x: x[:-1]
        testBad(b'\x00\x00\x00\x08\x06BCDEFGHIJK')
        self.proto.currentEncryptions = oldEncryptions
        self.proto.incomingCompression = MockCompression()

        def stubDecompress(payload):
            raise Exception('bad compression')
        self.proto.incomingCompression.decompress = stubDecompress
        testBad(b'\x00\x00\x00\x04\x00BCDE', transport.DISCONNECT_COMPRESSION_ERROR)
        self.flushLoggedErrors()

    def test_unimplementedPackets(self):
        """
        Test that unimplemented packet types cause MSG_UNIMPLEMENTED packets
        to be sent.
        """
        seqnum = self.proto.incomingPacketSequence

        def checkUnimplemented(seqnum=seqnum):
            self.assertEqual(self.packets[0][0], transport.MSG_UNIMPLEMENTED)
            self.assertEqual(self.packets[0][1][3:4], bytes((seqnum,)))
            self.proto.packets = []
            seqnum += 1
        self.proto.dispatchMessage(40, b'')
        checkUnimplemented()
        transport.messages[41] = b'MSG_fiction'
        self.proto.dispatchMessage(41, b'')
        checkUnimplemented()
        self.proto.dispatchMessage(60, b'')
        checkUnimplemented()
        self.proto.setService(MockService())
        self.proto.dispatchMessage(70, b'')
        checkUnimplemented()
        self.proto.dispatchMessage(71, b'')
        checkUnimplemented()

    def test_multipleClasses(self):
        """
        Test that multiple instances have distinct states.
        """
        proto = self.proto
        proto.dataReceived(self.transport.value())
        proto.currentEncryptions = MockCipher()
        proto.outgoingCompression = MockCompression()
        proto.incomingCompression = MockCompression()
        proto.setService(MockService())
        proto2 = MockTransportBase()
        proto2.makeConnection(proto_helpers.StringTransport())
        proto2.sendIgnore(b'')
        self.assertNotEqual(proto.gotVersion, proto2.gotVersion)
        self.assertNotEqual(proto.transport, proto2.transport)
        self.assertNotEqual(proto.outgoingPacketSequence, proto2.outgoingPacketSequence)
        self.assertNotEqual(proto.incomingPacketSequence, proto2.incomingPacketSequence)
        self.assertNotEqual(proto.currentEncryptions, proto2.currentEncryptions)
        self.assertNotEqual(proto.service, proto2.service)