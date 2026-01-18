import os.path
from errno import ENOSYS
from struct import pack
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
import hamcrest
from twisted.conch.error import ConchError, HostKeyChanged, UserRejectedKey
from twisted.conch.interfaces import IConchUser
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.portal import Portal
from twisted.internet.address import IPv4Address
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import (
from twisted.internet.interfaces import IAddress, IStreamClientEndpoint
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import (
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
from twisted.test.iosim import FakeTransport, connect
def test_readExisting(self):
    """
        Existing entries in the I{known_hosts} file are reflected by the
        L{KnownHostsFile} created by L{_NewConnectionHelper} when none is
        supplied to it.
        """
    key = CommandFactory().publicKeys[b'ssh-rsa']
    path = FilePath(self.mktemp())
    knownHosts = KnownHostsFile(path)
    knownHosts.addHostKey(b'127.0.0.1', key)
    knownHosts.save()
    msg(f'Created known_hosts file at {path.path!r}')
    home = os.path.expanduser('~/')
    default = path.path.replace(home, '~/')
    self.patch(_NewConnectionHelper, '_KNOWN_HOSTS', default)
    msg(f'Patched _KNOWN_HOSTS with {default!r}')
    loaded = _NewConnectionHelper._knownHosts()
    self.assertTrue(loaded.hasHostKey(b'127.0.0.1', key))