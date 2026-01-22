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
class _UserAuth(SSHUserAuthClient):
    """
    L{_UserAuth} implements the client part of SSH user authentication in the
    convenient way a user might expect if they are familiar with the
    interactive I{ssh} command line client.

    L{_UserAuth} supports key-based authentication, password-based
    authentication, and delegating authentication to an agent.
    """
    password = None
    keys = None
    agent = None

    def getPublicKey(self):
        """
        Retrieve the next public key object to offer to the server, possibly
        delegating to an authentication agent if there is one.

        @return: The public part of a key pair that could be used to
            authenticate with the server, or L{None} if there are no more
            public keys to try.
        @rtype: L{twisted.conch.ssh.keys.Key} or L{None}
        """
        if self.agent is not None:
            return self.agent.getPublicKey()
        if self.keys:
            self.key = self.keys.pop(0)
        else:
            self.key = None
        return self.key.public()

    def signData(self, publicKey, signData):
        """
        Extend the base signing behavior by using an SSH agent to sign the
        data, if one is available.

        @type publicKey: L{Key}
        @type signData: L{str}
        """
        if self.agent is not None:
            return self.agent.signData(publicKey.blob(), signData)
        else:
            return SSHUserAuthClient.signData(self, publicKey, signData)

    def getPrivateKey(self):
        """
        Get the private part of a key pair to use for authentication.  The key
        corresponds to the public part most recently returned from
        C{getPublicKey}.

        @return: A L{Deferred} which fires with the private key.
        @rtype: L{Deferred}
        """
        return succeed(self.key)

    def getPassword(self):
        """
        Get the password to use for authentication.

        @return: A L{Deferred} which fires with the password, or L{None} if the
            password was not specified.
        """
        if self.password is None:
            return
        return succeed(self.password)

    def ssh_USERAUTH_SUCCESS(self, packet):
        """
        Handle user authentication success in the normal way, but also make a
        note of the state change on the L{_CommandTransport}.
        """
        self.transport._state = b'CHANNELLING'
        return SSHUserAuthClient.ssh_USERAUTH_SUCCESS(self, packet)

    def connectToAgent(self, endpoint):
        """
        Set up a connection to the authentication agent and trigger its
        initialization.

        @param endpoint: An endpoint which can be used to connect to the
            authentication agent.
        @type endpoint: L{IStreamClientEndpoint} provider

        @return: A L{Deferred} which fires when the agent connection is ready
            for use.
        """
        factory = Factory()
        factory.protocol = SSHAgentClient
        d = endpoint.connect(factory)

        def connected(agent):
            self.agent = agent
            return agent.getPublicKeys()
        d.addCallback(connected)
        return d

    def loseAgentConnection(self):
        """
        Disconnect the agent.
        """
        if self.agent is None:
            return
        self.agent.transport.loseConnection()