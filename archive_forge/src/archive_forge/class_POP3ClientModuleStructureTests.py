import inspect
import sys
from typing import List
from unittest import skipIf
from zope.interface import directlyProvides
import twisted.mail._pop3client
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.mail.pop3 import (
from twisted.mail.test import pop3testserver
from twisted.protocols import basic, loopback
from twisted.python import log
from twisted.trial.unittest import TestCase
class POP3ClientModuleStructureTests(TestCase):
    """
    Miscellaneous tests more to do with module/package structure than
    anything to do with the POP3 client.
    """

    def test_all(self):
        """
        twisted.mail._pop3client.__all__ should be empty because all classes
        should be imported through twisted.mail.pop3.
        """
        self.assertEqual(twisted.mail._pop3client.__all__, [])

    def test_import(self):
        """
        Every public class in twisted.mail._pop3client should be available as
        a member of twisted.mail.pop3 with the exception of
        twisted.mail._pop3client.POP3Client which should be available as
        twisted.mail.pop3.AdvancedClient.
        """
        publicClasses = [c[0] for c in inspect.getmembers(sys.modules['twisted.mail._pop3client'], inspect.isclass) if not c[0][0] == '_']
        for pc in publicClasses:
            if not pc == 'POP3Client':
                self.assertTrue(hasattr(twisted.mail.pop3, pc), f'{pc} not in {twisted.mail.pop3}')
            else:
                self.assertTrue(hasattr(twisted.mail.pop3, 'AdvancedPOP3Client'))