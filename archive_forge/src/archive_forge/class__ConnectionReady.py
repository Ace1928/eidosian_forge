import signal
from os.path import expanduser
from struct import unpack
from zope.interface import Interface, implementer
from twisted.conch.client.agent import SSHAgentClient
from twisted.conch.client.default import _KNOWN_HOSTS
from twisted.conch.client.knownhosts import ConsoleUI, KnownHostsFile
from twisted.conch.ssh.channel import SSHChannel
from twisted.conch.ssh.common import NS, getNS
from twisted.conch.ssh.connection import SSHConnection
from twisted.conch.ssh.keys import Key
from twisted.conch.ssh.transport import SSHClientTransport
from twisted.conch.ssh.userauth import SSHUserAuthClient
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol
from twisted.internet.error import ConnectionDone, ProcessTerminated
from twisted.internet.interfaces import IStreamClientEndpoint
from twisted.internet.protocol import Factory
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
class _ConnectionReady(SSHConnection):
    """
    L{_ConnectionReady} is an L{SSHConnection} (an SSH service) which only
    propagates the I{serviceStarted} event to a L{Deferred} to be handled
    elsewhere.
    """

    def __init__(self, ready):
        """
        @param ready: A L{Deferred} which should be fired when
            I{serviceStarted} happens.
        """
        SSHConnection.__init__(self)
        self._ready = ready

    def serviceStarted(self):
        """
        When the SSH I{connection} I{service} this object represents is ready
        to be used, fire the C{connectionReady} L{Deferred} to publish that
        event to some other interested party.

        """
        self._ready.callback(self)
        del self._ready