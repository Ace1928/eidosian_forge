from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
class ParseTests(unittest.TestCase):

    def testParseAddress(self):
        for address, name, urls, params in [('"A. G. Bell" <sip:foo@example.com>', 'A. G. Bell', 'sip:foo@example.com', {}), ('Anon <sip:foo@example.com>', 'Anon', 'sip:foo@example.com', {}), ('sip:foo@example.com', '', 'sip:foo@example.com', {}), ('<sip:foo@example.com>', '', 'sip:foo@example.com', {}), ('foo <sip:foo@example.com>;tag=bar;foo=baz', 'foo', 'sip:foo@example.com', {'tag': 'bar', 'foo': 'baz'})]:
            gname, gurl, gparams = sip.parseAddress(address)
            self.assertEqual(name, gname)
            self.assertEqual(gurl.toString(), urls)
            self.assertEqual(gparams, params)