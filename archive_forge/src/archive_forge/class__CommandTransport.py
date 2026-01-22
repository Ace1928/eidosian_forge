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
class _CommandTransport(SSHClientTransport):
    """
    L{_CommandTransport} is an SSH client I{transport} which includes a host
    key verification step before it will proceed to secure the connection.

    L{_CommandTransport} also knows how to set up a connection to an
    authentication agent if it is told where it can connect to one.

    @ivar _userauth: The L{_UserAuth} instance which is in charge of the
        overall authentication process or L{None} if the SSH connection has not
        reach yet the C{user-auth} service.
    @type _userauth: L{_UserAuth}
    """
    _state = b'STARTING'
    _hostKeyFailure = None
    _userauth = None

    def __init__(self, creator):
        """
        @param creator: The L{_NewConnectionHelper} that created this
            connection.

        @type creator: L{_NewConnectionHelper}.
        """
        self.connectionReady = Deferred(lambda d: self.transport.abortConnection())

        def readyFired(result):
            self.connectionReady = None
            return result
        self.connectionReady.addBoth(readyFired)
        self.creator = creator

    def verifyHostKey(self, hostKey, fingerprint):
        """
        Ask the L{KnownHostsFile} provider available on the factory which
        created this protocol this protocol to verify the given host key.

        @return: A L{Deferred} which fires with the result of
            L{KnownHostsFile.verifyHostKey}.
        """
        hostname = self.creator.hostname
        ip = networkString(self.transport.getPeer().host)
        self._state = b'SECURING'
        d = self.creator.knownHosts.verifyHostKey(self.creator.ui, hostname, ip, Key.fromString(hostKey))
        d.addErrback(self._saveHostKeyFailure)
        return d

    def _saveHostKeyFailure(self, reason):
        """
        When host key verification fails, record the reason for the failure in
        order to fire a L{Deferred} with it later.

        @param reason: The cause of the host key verification failure.
        @type reason: L{Failure}

        @return: C{reason}
        @rtype: L{Failure}
        """
        self._hostKeyFailure = reason
        return reason

    def connectionSecure(self):
        """
        When the connection is secure, start the authentication process.
        """
        self._state = b'AUTHENTICATING'
        command = _ConnectionReady(self.connectionReady)
        self._userauth = _UserAuth(self.creator.username, command)
        self._userauth.password = self.creator.password
        if self.creator.keys:
            self._userauth.keys = list(self.creator.keys)
        if self.creator.agentEndpoint is not None:
            d = self._userauth.connectToAgent(self.creator.agentEndpoint)
        else:
            d = succeed(None)

        def maybeGotAgent(ignored):
            self.requestService(self._userauth)
        d.addBoth(maybeGotAgent)

    def connectionLost(self, reason):
        """
        When the underlying connection to the SSH server is lost, if there were
        any connection setup errors, propagate them. Also, clean up the
        connection to the ssh agent if one was created.
        """
        if self._userauth:
            self._userauth.loseAgentConnection()
        if self._state == b'RUNNING' or self.connectionReady is None:
            return
        if self._state == b'SECURING' and self._hostKeyFailure is not None:
            reason = self._hostKeyFailure
        elif self._state == b'AUTHENTICATING':
            reason = Failure(AuthenticationFailed('Connection lost while authenticating'))
        self.connectionReady.errback(reason)