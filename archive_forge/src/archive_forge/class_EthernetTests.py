from zope.interface import implementer
from twisted.pair import ethernet, raw
from twisted.python import components
from twisted.trial import unittest
class EthernetTests(unittest.TestCase):

    def testPacketParsing(self):
        proto = ethernet.EthernetProtocol()
        p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': b'123456', 'source': b'989284', 'protocol': 2048})])
        proto.addProto(2048, p1)
        proto.datagramReceived(b'123456989284\x08\x00foobar', partial=0)
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting

    def testMultiplePackets(self):
        proto = ethernet.EthernetProtocol()
        p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': b'123456', 'source': b'989284', 'protocol': 2048}), (b'quux', {'partial': 1, 'dest': b'012345', 'source': b'abcdef', 'protocol': 2048})])
        proto.addProto(2048, p1)
        proto.datagramReceived(b'123456989284\x08\x00foobar', partial=0)
        proto.datagramReceived(b'012345abcdef\x08\x00quux', partial=1)
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting

    def testMultipleSameProtos(self):
        proto = ethernet.EthernetProtocol()
        p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': b'123456', 'source': b'989284', 'protocol': 2048})])
        p2 = MyProtocol([(b'foobar', {'partial': 0, 'dest': b'123456', 'source': b'989284', 'protocol': 2048})])
        proto.addProto(2048, p1)
        proto.addProto(2048, p2)
        proto.datagramReceived(b'123456989284\x08\x00foobar', partial=0)
        assert not p1.expecting, 'Should not expect any more packets, but still want {!r}'.format(p1.expecting)
        assert not p2.expecting, 'Should not expect any more packets, but still want {!r}'.format(p2.expecting)

    def testWrongProtoNotSeen(self):
        proto = ethernet.EthernetProtocol()
        p1 = MyProtocol([])
        proto.addProto(2049, p1)
        proto.datagramReceived(b'123456989284\x08\x00foobar', partial=0)
        proto.datagramReceived(b'012345abcdef\x08\x00quux', partial=1)

    def testDemuxing(self):
        proto = ethernet.EthernetProtocol()
        p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': b'123456', 'source': b'989284', 'protocol': 2048}), (b'quux', {'partial': 1, 'dest': b'012345', 'source': b'abcdef', 'protocol': 2048})])
        proto.addProto(2048, p1)
        p2 = MyProtocol([(b'quux', {'partial': 1, 'dest': b'012345', 'source': b'abcdef', 'protocol': 2054}), (b'foobar', {'partial': 0, 'dest': b'123456', 'source': b'989284', 'protocol': 2054})])
        proto.addProto(2054, p2)
        proto.datagramReceived(b'123456989284\x08\x00foobar', partial=0)
        proto.datagramReceived(b'012345abcdef\x08\x06quux', partial=1)
        proto.datagramReceived(b'123456989284\x08\x06foobar', partial=0)
        proto.datagramReceived(b'012345abcdef\x08\x00quux', partial=1)
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting
        assert not p2.expecting, 'Should not expect any more packets, but still want %r' % p2.expecting

    def testAddingBadProtos_WrongLevel(self):
        """Adding a wrong level protocol raises an exception."""
        e = ethernet.EthernetProtocol()
        try:
            e.addProto(42, 'silliness')
        except components.CannotAdapt:
            pass
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')

    def testAddingBadProtos_TooSmall(self):
        """Adding a protocol with a negative number raises an exception."""
        e = ethernet.EthernetProtocol()
        try:
            e.addProto(-1, MyProtocol([]))
        except TypeError as e:
            if e.args == ('Added protocol must be positive or zero',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')

    def testAddingBadProtos_TooBig(self):
        """Adding a protocol with a number >=2**16 raises an exception."""
        e = ethernet.EthernetProtocol()
        try:
            e.addProto(2 ** 16, MyProtocol([]))
        except TypeError as e:
            if e.args == ('Added protocol must fit in 16 bits',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')

    def testAddingBadProtos_TooBig2(self):
        """Adding a protocol with a number >=2**16 raises an exception."""
        e = ethernet.EthernetProtocol()
        try:
            e.addProto(2 ** 16 + 1, MyProtocol([]))
        except TypeError as e:
            if e.args == ('Added protocol must fit in 16 bits',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')