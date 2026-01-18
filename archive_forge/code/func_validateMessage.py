from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def validateMessage(self, m, method, uri, headers, body):
    """
        Validate Requests.
        """
    self.assertEqual(m.method, method)
    self.assertEqual(m.uri.toString(), uri)
    self.assertEqual(m.headers, headers)
    self.assertEqual(m.body, body)
    self.assertEqual(m.finished, 1)