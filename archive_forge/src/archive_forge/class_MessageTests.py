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
class MessageTests(unittest.SynchronousTestCase):
    """
    Tests for L{twisted.names.dns.Message}.
    """

    def test_authenticDataDefault(self):
        """
        L{dns.Message.authenticData} has default value 0.
        """
        self.assertEqual(dns.Message().authenticData, 0)

    def test_authenticDataOverride(self):
        """
        L{dns.Message.__init__} accepts a C{authenticData} argument which
        is assigned to L{dns.Message.authenticData}.
        """
        self.assertEqual(dns.Message(authenticData=1).authenticData, 1)

    def test_authenticDataEncode(self):
        """
        L{dns.Message.toStr} encodes L{dns.Message.authenticData} into
        byte4 of the byte string.
        """
        self.assertEqual(dns.Message(authenticData=1).toStr(), MESSAGE_AUTHENTIC_DATA_BYTES)

    def test_authenticDataDecode(self):
        """
        L{dns.Message.fromStr} decodes byte4 and assigns bit3 to
        L{dns.Message.authenticData}.
        """
        m = dns.Message()
        m.fromStr(MESSAGE_AUTHENTIC_DATA_BYTES)
        self.assertEqual(m.authenticData, 1)

    def test_checkingDisabledDefault(self):
        """
        L{dns.Message.checkingDisabled} has default value 0.
        """
        self.assertEqual(dns.Message().checkingDisabled, 0)

    def test_checkingDisabledOverride(self):
        """
        L{dns.Message.__init__} accepts a C{checkingDisabled} argument which
        is assigned to L{dns.Message.checkingDisabled}.
        """
        self.assertEqual(dns.Message(checkingDisabled=1).checkingDisabled, 1)

    def test_checkingDisabledEncode(self):
        """
        L{dns.Message.toStr} encodes L{dns.Message.checkingDisabled} into
        byte4 of the byte string.
        """
        self.assertEqual(dns.Message(checkingDisabled=1).toStr(), MESSAGE_CHECKING_DISABLED_BYTES)

    def test_checkingDisabledDecode(self):
        """
        L{dns.Message.fromStr} decodes byte4 and assigns bit4 to
        L{dns.Message.checkingDisabled}.
        """
        m = dns.Message()
        m.fromStr(MESSAGE_CHECKING_DISABLED_BYTES)
        self.assertEqual(m.checkingDisabled, 1)

    def test_reprDefaults(self):
        """
        L{dns.Message.__repr__} omits field values and sections which are
        identical to their defaults. The id field value is always shown.
        """
        self.assertEqual('<Message id=0>', repr(dns.Message()))

    def test_reprFlagsIfSet(self):
        """
        L{dns.Message.__repr__} displays flags if they are L{True}.
        """
        m = dns.Message(answer=True, auth=True, trunc=True, recDes=True, recAv=True, authenticData=True, checkingDisabled=True)
        self.assertEqual('<Message id=0 flags=answer,auth,trunc,recDes,recAv,authenticData,checkingDisabled>', repr(m))

    def test_reprNonDefautFields(self):
        """
        L{dns.Message.__repr__} displays field values if they differ from their
        defaults.
        """
        m = dns.Message(id=10, opCode=20, rCode=30, maxSize=40)
        self.assertEqual('<Message id=10 opCode=20 rCode=30 maxSize=40>', repr(m))

    def test_reprNonDefaultSections(self):
        """
        L{dns.Message.__repr__} displays sections which differ from their
        defaults.
        """
        m = dns.Message()
        m.queries = [1, 2, 3]
        m.answers = [4, 5, 6]
        m.authority = [7, 8, 9]
        m.additional = [10, 11, 12]
        self.assertEqual('<Message id=0 queries=[1, 2, 3] answers=[4, 5, 6] authority=[7, 8, 9] additional=[10, 11, 12]>', repr(m))

    def test_emptyMessage(self):
        """
        Test that a message which has been truncated causes an EOFError to
        be raised when it is parsed.
        """
        msg = dns.Message()
        self.assertRaises(EOFError, msg.fromStr, b'')

    def test_emptyQuery(self):
        """
        Test that bytes representing an empty query message can be decoded
        as such.
        """
        msg = dns.Message()
        msg.fromStr(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
        self.assertEqual(msg.id, 256)
        self.assertFalse(msg.answer, 'Message was not supposed to be an answer.')
        self.assertEqual(msg.opCode, dns.OP_QUERY)
        self.assertFalse(msg.auth, 'Message was not supposed to be authoritative.')
        self.assertFalse(msg.trunc, 'Message was not supposed to be truncated.')
        self.assertEqual(msg.queries, [])
        self.assertEqual(msg.answers, [])
        self.assertEqual(msg.authority, [])
        self.assertEqual(msg.additional, [])

    def test_NULL(self):
        """
        A I{NULL} record with an arbitrary payload can be encoded and decoded as
        part of a L{dns.Message}.
        """
        bytes = b''.join([dns._ord2bytes(i) for i in range(256)])
        rec = dns.Record_NULL(bytes)
        rr = dns.RRHeader(b'testname', dns.NULL, payload=rec)
        msg1 = dns.Message()
        msg1.answers.append(rr)
        s = BytesIO()
        msg1.encode(s)
        s.seek(0, 0)
        msg2 = dns.Message()
        msg2.decode(s)
        self.assertIsInstance(msg2.answers[0].payload, dns.Record_NULL)
        self.assertEqual(msg2.answers[0].payload.payload, bytes)

    def test_lookupRecordTypeDefault(self):
        """
        L{Message.lookupRecordType} returns C{dns.UnknownRecord} if it is
        called with an integer which doesn't correspond to any known record
        type.
        """
        self.assertIs(dns.Message().lookupRecordType(65280), dns.UnknownRecord)

    def test_nonAuthoritativeMessage(self):
        """
        The L{RRHeader} instances created by L{Message} from a non-authoritative
        message are marked as not authoritative.
        """
        buf = BytesIO()
        answer = dns.RRHeader(payload=dns.Record_A('1.2.3.4', ttl=0))
        answer.encode(buf)
        message = dns.Message()
        message.fromStr(b'\x01\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00' + buf.getvalue())
        self.assertEqual(message.answers, [answer])
        self.assertFalse(message.answers[0].auth)

    def test_authoritativeMessage(self):
        """
        The L{RRHeader} instances created by L{Message} from an authoritative
        message are marked as authoritative.
        """
        buf = BytesIO()
        answer = dns.RRHeader(payload=dns.Record_A('1.2.3.4', ttl=0))
        answer.encode(buf)
        message = dns.Message()
        message.fromStr(b'\x01\x00\x04\x00\x00\x00\x00\x01\x00\x00\x00\x00' + buf.getvalue())
        answer.auth = True
        self.assertEqual(message.answers, [answer])
        self.assertTrue(message.answers[0].auth)