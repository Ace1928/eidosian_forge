import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
class EDNSMessageEDNSEncodingTests(unittest.SynchronousTestCase):
    """
    Tests for the encoding and decoding of various EDNS messages.

    These test will not work with L{dns.Message}.
    """
    messageFactory = dns._EDNSMessage

    def test_ednsMessageDecodeStripsOptRecords(self):
        """
        The L(_EDNSMessage} instance created by L{dns._EDNSMessage.decode} from
        an EDNS query never includes OPT records in the additional section.
        """
        m = self.messageFactory()
        m.fromStr(MessageEDNSQuery.bytes())
        self.assertEqual(m.additional, [])

    def test_ednsMessageDecodeMultipleOptRecords(self):
        """
        An L(_EDNSMessage} instance created from a byte string containing
        multiple I{OPT} records will discard all the C{OPT} records.

        C{ednsVersion} will be set to L{None}.

        @see: U{https://tools.ietf.org/html/rfc6891#section-6.1.1}
        """
        m = dns.Message()
        m.additional = [dns._OPTHeader(version=2), dns._OPTHeader(version=3)]
        ednsMessage = dns._EDNSMessage()
        ednsMessage.fromStr(m.toStr())
        self.assertIsNone(ednsMessage.ednsVersion)

    def test_fromMessageCopiesSections(self):
        """
        L{dns._EDNSMessage._fromMessage} returns an L{_EDNSMessage} instance
        whose queries, answers, authority and additional lists are copies (not
        references to) the original message lists.
        """
        standardMessage = dns.Message()
        standardMessage.fromStr(MessageEDNSQuery.bytes())
        ednsMessage = dns._EDNSMessage._fromMessage(standardMessage)
        duplicates = []
        for attrName in ('queries', 'answers', 'authority', 'additional'):
            if getattr(standardMessage, attrName) is getattr(ednsMessage, attrName):
                duplicates.append(attrName)
        if duplicates:
            self.fail('Message and _EDNSMessage shared references to the following section lists after decoding: %s' % (duplicates,))

    def test_toMessageCopiesSections(self):
        """
        L{dns._EDNSMessage.toStr} makes no in place changes to the message
        instance.
        """
        ednsMessage = dns._EDNSMessage(ednsVersion=1)
        ednsMessage.toStr()
        self.assertEqual(ednsMessage.additional, [])

    def test_optHeaderPosition(self):
        """
        L{dns._EDNSMessage} can decode OPT records, regardless of their position
        in the additional records section.

        "The OPT RR MAY be placed anywhere within the additional data section."

        @see: U{https://tools.ietf.org/html/rfc6891#section-6.1.1}
        """
        b = BytesIO()
        optRecord = dns._OPTHeader(version=1)
        optRecord.encode(b)
        optRRHeader = dns.RRHeader()
        b.seek(0)
        optRRHeader.decode(b)
        m = dns.Message()
        m.additional = [optRRHeader]
        actualMessages = []
        actualMessages.append(dns._EDNSMessage._fromMessage(m).ednsVersion)
        m.additional.append(dns.RRHeader(type=dns.A))
        actualMessages.append(dns._EDNSMessage._fromMessage(m).ednsVersion)
        m.additional.insert(0, dns.RRHeader(type=dns.A))
        actualMessages.append(dns._EDNSMessage._fromMessage(m).ednsVersion)
        self.assertEqual([1] * 3, actualMessages)

    def test_ednsDecode(self):
        """
        The L(_EDNSMessage} instance created by L{dns._EDNSMessage.fromStr}
        derives its edns specific values (C{ednsVersion}, etc) from the supplied
        OPT record.
        """
        m = self.messageFactory()
        m.fromStr(MessageEDNSComplete.bytes())
        self.assertEqual(m, self.messageFactory(**MessageEDNSComplete.kwargs()))

    def test_ednsEncode(self):
        """
        The L(_EDNSMessage} instance created by L{dns._EDNSMessage.toStr}
        encodes its edns specific values (C{ednsVersion}, etc) into an OPT
        record added to the additional section.
        """
        self.assertEqual(self.messageFactory(**MessageEDNSComplete.kwargs()).toStr(), MessageEDNSComplete.bytes())

    def test_extendedRcodeEncode(self):
        """
        The L(_EDNSMessage.toStr} encodes the extended I{RCODE} (>=16) by
        assigning the lower 4bits to the message RCODE field and the upper 4bits
        to the OPT pseudo record.
        """
        self.assertEqual(self.messageFactory(**MessageEDNSExtendedRCODE.kwargs()).toStr(), MessageEDNSExtendedRCODE.bytes())

    def test_extendedRcodeDecode(self):
        """
        The L(_EDNSMessage} instance created by L{dns._EDNSMessage.fromStr}
        derives RCODE from the supplied OPT record.
        """
        m = self.messageFactory()
        m.fromStr(MessageEDNSExtendedRCODE.bytes())
        self.assertEqual(m, self.messageFactory(**MessageEDNSExtendedRCODE.kwargs()))

    def test_extendedRcodeZero(self):
        """
        Note that EXTENDED-RCODE value 0 indicates that an unextended RCODE is
        in use (values 0 through 15).

        https://tools.ietf.org/html/rfc6891#section-6.1.3
        """
        ednsMessage = self.messageFactory(rCode=15, ednsVersion=0)
        standardMessage = ednsMessage._toMessage()
        self.assertEqual((15, 0), (standardMessage.rCode, standardMessage.additional[0].extendedRCODE))