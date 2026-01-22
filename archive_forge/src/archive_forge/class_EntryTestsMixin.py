import os
from binascii import Error as BinasciiError, a2b_base64, b2a_base64
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.internet.defer import Deferred
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial.unittest import TestCase
class EntryTestsMixin:
    """
    Tests for implementations of L{IKnownHostEntry}.  Subclasses must set the
    'entry' attribute to a provider of that interface, the implementation of
    that interface under test.

    @ivar entry: a provider of L{IKnownHostEntry} with a hostname of
    www.twistedmatrix.com and an RSA key of sampleKey.
    """

    def test_providesInterface(self):
        """
        The given entry should provide IKnownHostEntry.
        """
        verifyObject(IKnownHostEntry, self.entry)

    def test_fromString(self):
        """
        Constructing a plain text entry from an unhashed known_hosts entry will
        result in an L{IKnownHostEntry} provider with 'keyString', 'hostname',
        and 'keyType' attributes.  While outside the interface in question,
        these attributes are held in common by L{PlainEntry} and L{HashedEntry}
        implementations; other implementations should override this method in
        subclasses.
        """
        entry = self.entry
        self.assertEqual(entry.publicKey, Key.fromString(sampleKey))
        self.assertEqual(entry.keyType, b'ssh-rsa')

    def test_matchesKey(self):
        """
        L{IKnownHostEntry.matchesKey} checks to see if an entry matches a given
        SSH key.
        """
        twistedmatrixDotCom = Key.fromString(sampleKey)
        divmodDotCom = Key.fromString(otherSampleKey)
        self.assertEqual(True, self.entry.matchesKey(twistedmatrixDotCom))
        self.assertEqual(False, self.entry.matchesKey(divmodDotCom))

    def test_matchesHost(self):
        """
        L{IKnownHostEntry.matchesHost} checks to see if an entry matches a
        given hostname.
        """
        self.assertTrue(self.entry.matchesHost(b'www.twistedmatrix.com'))
        self.assertFalse(self.entry.matchesHost(b'www.divmod.com'))