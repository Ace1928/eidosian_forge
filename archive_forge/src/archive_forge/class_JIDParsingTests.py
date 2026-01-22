from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
class JIDParsingTests(unittest.TestCase):

    def test_parse(self) -> None:
        """
        Test different forms of JIDs.
        """
        self.assertEqual(jid.parse('user@host/resource'), ('user', 'host', 'resource'))
        self.assertEqual(jid.parse('user@host'), ('user', 'host', None))
        self.assertEqual(jid.parse('host'), (None, 'host', None))
        self.assertEqual(jid.parse('host/resource'), (None, 'host', 'resource'))
        self.assertEqual(jid.parse('foo/bar@baz'), (None, 'foo', 'bar@baz'))
        self.assertEqual(jid.parse('boo@foo/bar@baz'), ('boo', 'foo', 'bar@baz'))
        self.assertEqual(jid.parse('boo@foo/bar/baz'), ('boo', 'foo', 'bar/baz'))
        self.assertEqual(jid.parse('boo/foo@bar@baz'), (None, 'boo', 'foo@bar@baz'))
        self.assertEqual(jid.parse('boo/foo/bar'), (None, 'boo', 'foo/bar'))
        self.assertEqual(jid.parse('boo//foo'), (None, 'boo', '/foo'))

    def test_noHost(self) -> None:
        """
        Test for failure on no host part.
        """
        self.assertRaises(jid.InvalidFormat, jid.parse, 'user@')

    def test_doubleAt(self) -> None:
        """
        Test for failure on double @ signs.

        This should fail because @ is not a valid character for the host
        part of the JID.
        """
        self.assertRaises(jid.InvalidFormat, jid.parse, 'user@@host')

    def test_multipleAt(self) -> None:
        """
        Test for failure on two @ signs.

        This should fail because @ is not a valid character for the host
        part of the JID.
        """
        self.assertRaises(jid.InvalidFormat, jid.parse, 'user@host@host')

    def test_prepCaseMapUser(self) -> None:
        """
        Test case mapping of the user part of the JID.
        """
        self.assertEqual(jid.prep('UsEr', 'host', 'resource'), ('user', 'host', 'resource'))

    def test_prepCaseMapHost(self) -> None:
        """
        Test case mapping of the host part of the JID.
        """
        self.assertEqual(jid.prep('user', 'hoST', 'resource'), ('user', 'host', 'resource'))

    def test_prepNoCaseMapResource(self) -> None:
        """
        Test no case mapping of the resourcce part of the JID.
        """
        self.assertEqual(jid.prep('user', 'hoST', 'resource'), ('user', 'host', 'resource'))
        self.assertNotEqual(jid.prep('user', 'host', 'Resource'), ('user', 'host', 'resource'))