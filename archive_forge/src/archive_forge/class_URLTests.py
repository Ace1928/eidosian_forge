from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
class URLTests(unittest.TestCase):

    def testRoundtrip(self):
        for url in ['sip:j.doe@big.com', 'sip:j.doe:secret@big.com;transport=tcp', 'sip:j.doe@big.com?subject=project', 'sip:example.com']:
            self.assertEqual(sip.parseURL(url).toString(), url)

    def testComplex(self):
        s = 'sip:user:pass@hosta:123;transport=udp;user=phone;method=foo;ttl=12;maddr=1.2.3.4;blah;goo=bar?a=b&c=d'
        url = sip.parseURL(s)
        for k, v in [('username', 'user'), ('password', 'pass'), ('host', 'hosta'), ('port', 123), ('transport', 'udp'), ('usertype', 'phone'), ('method', 'foo'), ('ttl', 12), ('maddr', '1.2.3.4'), ('other', ['blah', 'goo=bar']), ('headers', {'a': 'b', 'c': 'd'})]:
            self.assertEqual(getattr(url, k), v)