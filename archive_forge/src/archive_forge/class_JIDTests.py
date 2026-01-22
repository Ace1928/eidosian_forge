from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
class JIDTests(unittest.TestCase):

    def test_noneArguments(self) -> None:
        """
        Test that using no arguments raises an exception.
        """
        self.assertRaises(RuntimeError, jid.JID)

    def test_attributes(self) -> None:
        """
        Test that the attributes correspond with the JID parts.
        """
        j = jid.JID('user@host/resource')
        self.assertEqual(j.user, 'user')
        self.assertEqual(j.host, 'host')
        self.assertEqual(j.resource, 'resource')

    def test_userhost(self) -> None:
        """
        Test the extraction of the bare JID.
        """
        j = jid.JID('user@host/resource')
        self.assertEqual('user@host', j.userhost())

    def test_userhostOnlyHost(self) -> None:
        """
        Test the extraction of the bare JID of the full form host/resource.
        """
        j = jid.JID('host/resource')
        self.assertEqual('host', j.userhost())

    def test_userhostJID(self) -> None:
        """
        Test getting a JID object of the bare JID.
        """
        j1 = jid.JID('user@host/resource')
        j2 = jid.internJID('user@host')
        self.assertIdentical(j2, j1.userhostJID())

    def test_userhostJIDNoResource(self) -> None:
        """
        Test getting a JID object of the bare JID when there was no resource.
        """
        j = jid.JID('user@host')
        self.assertIdentical(j, j.userhostJID())

    def test_fullHost(self) -> None:
        """
        Test giving a string representation of the JID with only a host part.
        """
        j = jid.JID(tuple=(None, 'host', None))
        self.assertEqual('host', j.full())

    def test_fullHostResource(self) -> None:
        """
        Test giving a string representation of the JID with host, resource.
        """
        j = jid.JID(tuple=(None, 'host', 'resource'))
        self.assertEqual('host/resource', j.full())

    def test_fullUserHost(self) -> None:
        """
        Test giving a string representation of the JID with user, host.
        """
        j = jid.JID(tuple=('user', 'host', None))
        self.assertEqual('user@host', j.full())

    def test_fullAll(self) -> None:
        """
        Test giving a string representation of the JID.
        """
        j = jid.JID(tuple=('user', 'host', 'resource'))
        self.assertEqual('user@host/resource', j.full())

    def test_equality(self) -> None:
        """
        Test JID equality.
        """
        j1 = jid.JID('user@host/resource')
        j2 = jid.JID('user@host/resource')
        self.assertNotIdentical(j1, j2)
        self.assertEqual(j1, j2)

    def test_equalityWithNonJIDs(self) -> None:
        """
        Test JID equality.
        """
        j = jid.JID('user@host/resource')
        self.assertFalse(j == 'user@host/resource')

    def test_inequality(self) -> None:
        """
        Test JID inequality.
        """
        j1 = jid.JID('user1@host/resource')
        j2 = jid.JID('user2@host/resource')
        self.assertNotEqual(j1, j2)

    def test_inequalityWithNonJIDs(self) -> None:
        """
        Test JID equality.
        """
        j = jid.JID('user@host/resource')
        self.assertNotEqual(j, 'user@host/resource')

    def test_hashable(self) -> None:
        """
        Test JID hashability.
        """
        j1 = jid.JID('user@host/resource')
        j2 = jid.JID('user@host/resource')
        self.assertEqual(hash(j1), hash(j2))

    def test_str(self) -> None:
        """
        Test unicode representation of JIDs.
        """
        j = jid.JID(tuple=('user', 'host', 'resource'))
        self.assertEqual('user@host/resource', str(j))

    def test_repr(self) -> None:
        """
        Test representation of JID objects.
        """
        j = jid.JID(tuple=('user', 'host', 'resource'))
        self.assertEqual('JID(%s)' % repr('user@host/resource'), repr(j))