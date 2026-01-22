from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def _assertFailures(self, failures, *expectedFailures):
    for flag, failure in failures:
        self.assertEqual(flag, defer.FAILURE)
        failure.trap(*expectedFailures)
    return None