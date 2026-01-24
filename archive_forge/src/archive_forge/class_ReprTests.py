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
class ReprTests(unittest.TestCase):
    """
    Tests for the C{__repr__} implementation of record classes.
    """

    def test_ns(self):
        """
        The repr of a L{dns.Record_NS} instance includes the name of the
        nameserver and the TTL of the record.
        """
        self.assertEqual(repr(dns.Record_NS(b'example.com', 4321)), '<NS name=example.com ttl=4321>')

    def test_md(self):
        """
        The repr of a L{dns.Record_MD} instance includes the name of the
        mail destination and the TTL of the record.
        """
        self.assertEqual(repr(dns.Record_MD(b'example.com', 4321)), '<MD name=example.com ttl=4321>')

    def test_mf(self):
        """
        The repr of a L{dns.Record_MF} instance includes the name of the
        mail forwarder and the TTL of the record.
        """
        self.assertEqual(repr(dns.Record_MF(b'example.com', 4321)), '<MF name=example.com ttl=4321>')

    def test_cname(self):
        """
        The repr of a L{dns.Record_CNAME} instance includes the name of the
        mail forwarder and the TTL of the record.
        """
        self.assertEqual(repr(dns.Record_CNAME(b'example.com', 4321)), '<CNAME name=example.com ttl=4321>')

    def test_mb(self):
        """
        The repr of a L{dns.Record_MB} instance includes the name of the
        mailbox and the TTL of the record.
        """
        self.assertEqual(repr(dns.Record_MB(b'example.com', 4321)), '<MB name=example.com ttl=4321>')

    def test_mg(self):
        """
        The repr of a L{dns.Record_MG} instance includes the name of the
        mail group member and the TTL of the record.
        """
        self.assertEqual(repr(dns.Record_MG(b'example.com', 4321)), '<MG name=example.com ttl=4321>')

    def test_mr(self):
        """
        The repr of a L{dns.Record_MR} instance includes the name of the
        mail rename domain and the TTL of the record.
        """
        self.assertEqual(repr(dns.Record_MR(b'example.com', 4321)), '<MR name=example.com ttl=4321>')

    def test_ptr(self):
        """
        The repr of a L{dns.Record_PTR} instance includes the name of the
        pointer and the TTL of the record.
        """
        self.assertEqual(repr(dns.Record_PTR(b'example.com', 4321)), '<PTR name=example.com ttl=4321>')

    def test_dname(self):
        """
        The repr of a L{dns.Record_DNAME} instance includes the name of the
        non-terminal DNS name redirection and the TTL of the record.
        """
        self.assertEqual(repr(dns.Record_DNAME(b'example.com', 4321)), '<DNAME name=example.com ttl=4321>')

    def test_a(self):
        """
        The repr of a L{dns.Record_A} instance includes the dotted-quad
        string representation of the address it is for and the TTL of the
        record.
        """
        self.assertEqual(repr(dns.Record_A('1.2.3.4', 567)), '<A address=1.2.3.4 ttl=567>')

    def test_soa(self):
        """
        The repr of a L{dns.Record_SOA} instance includes all of the
        authority fields.
        """
        self.assertEqual(repr(dns.Record_SOA(mname=b'mName', rname=b'rName', serial=123, refresh=456, retry=789, expire=10, minimum=11, ttl=12)), '<SOA mname=mName rname=rName serial=123 refresh=456 retry=789 expire=10 minimum=11 ttl=12>')

    def test_null(self):
        """
        The repr of a L{dns.Record_NULL} instance includes the repr of its
        payload and the TTL of the record.
        """
        self.assertEqual(repr(dns.Record_NULL(b'abcd', 123)), "<NULL payload='abcd' ttl=123>")

    def test_wks(self):
        """
        The repr of a L{dns.Record_WKS} instance includes the dotted-quad
        string representation of the address it is for, the IP protocol
        number it is for, and the TTL of the record.
        """
        self.assertEqual(repr(dns.Record_WKS('2.3.4.5', 7, ttl=8)), '<WKS address=2.3.4.5 protocol=7 ttl=8>')

    def test_aaaa(self):
        """
        The repr of a L{dns.Record_AAAA} instance includes the colon-separated
        hex string representation of the address it is for and the TTL of the
        record.
        """
        self.assertEqual(repr(dns.Record_AAAA('8928::1234', ttl=10)), '<AAAA address=8928::1234 ttl=10>')

    def test_a6(self):
        """
        The repr of a L{dns.Record_A6} instance includes the colon-separated
        hex string representation of the address it is for and the TTL of the
        record.
        """
        self.assertEqual(repr(dns.Record_A6(0, '1234::5678', b'foo.bar', ttl=10)), '<A6 suffix=1234::5678 prefix=foo.bar ttl=10>')

    def test_srv(self):
        """
        The repr of a L{dns.Record_SRV} instance includes the name and port of
        the target and the priority, weight, and TTL of the record.
        """
        self.assertEqual(repr(dns.Record_SRV(1, 2, 3, b'example.org', 4)), '<SRV priority=1 weight=2 target=example.org port=3 ttl=4>')

    def test_naptr(self):
        """
        The repr of a L{dns.Record_NAPTR} instance includes the order,
        preference, flags, service, regular expression, replacement, and TTL of
        the record.
        """
        record = dns.Record_NAPTR(5, 9, b'S', b'http', b'/foo/bar/i', b'baz', 3)
        self.assertEqual(repr(record), '<NAPTR order=5 preference=9 flags=S service=http regexp=/foo/bar/i replacement=baz ttl=3>')

    def test_afsdb(self):
        """
        The repr of a L{dns.Record_AFSDB} instance includes the subtype,
        hostname, and TTL of the record.
        """
        self.assertEqual(repr(dns.Record_AFSDB(3, b'example.org', 5)), '<AFSDB subtype=3 hostname=example.org ttl=5>')

    def test_rp(self):
        """
        The repr of a L{dns.Record_RP} instance includes the mbox, txt, and TTL
        fields of the record.
        """
        self.assertEqual(repr(dns.Record_RP(b'alice.example.com', b'admin.example.com', 3)), '<RP mbox=alice.example.com txt=admin.example.com ttl=3>')

    def test_hinfo(self):
        """
        The repr of a L{dns.Record_HINFO} instance includes the cpu, os, and
        TTL fields of the record.
        """
        self.assertEqual(repr(dns.Record_HINFO(b'sparc', b'minix', 12)), "<HINFO cpu='sparc' os='minix' ttl=12>")

    def test_minfo(self):
        """
        The repr of a L{dns.Record_MINFO} instance includes the rmailbx,
        emailbx, and TTL fields of the record.
        """
        record = dns.Record_MINFO(b'alice.example.com', b'bob.example.com', 15)
        self.assertEqual(repr(record), '<MINFO responsibility=alice.example.com errors=bob.example.com ttl=15>')

    def test_mx(self):
        """
        The repr of a L{dns.Record_MX} instance includes the preference, name,
        and TTL fields of the record.
        """
        self.assertEqual(repr(dns.Record_MX(13, b'mx.example.com', 2)), '<MX preference=13 name=mx.example.com ttl=2>')

    def test_txt(self):
        """
        The repr of a L{dns.Record_TXT} instance includes the data and ttl
        fields of the record.
        """
        self.assertEqual(repr(dns.Record_TXT(b'foo', b'bar', ttl=15)), "<TXT data=['foo', 'bar'] ttl=15>")

    def test_spf(self):
        """
        The repr of a L{dns.Record_SPF} instance includes the data and ttl
        fields of the record.
        """
        self.assertEqual(repr(dns.Record_SPF(b'foo', b'bar', ttl=15)), "<SPF data=['foo', 'bar'] ttl=15>")

    def test_unknown(self):
        """
        The repr of a L{dns.UnknownRecord} instance includes the data and ttl
        fields of the record.
        """
        self.assertEqual(repr(dns.UnknownRecord(b'foo\x1fbar', 12)), "<UNKNOWN data='foo\\x1fbar' ttl=12>")