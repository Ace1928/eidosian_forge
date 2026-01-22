import email.utils
import os
import pickle
import time
from typing import Type
from twisted.application import internet
from twisted.internet import protocol
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import DNSLookupError
from twisted.internet.protocol import connectionDone
from twisted.mail import bounce, relay, smtp
from twisted.python import log
from twisted.python.failure import Failure
class SMTPManagedRelayerFactory(protocol.ClientFactory):
    """
    A factory to create an L{SMTPManagedRelayer}.

    This factory creates a managed relayer which relays a set of messages over
    SMTP and informs an attempt manager of its progress.

    @ivar messages: See L{__init__}
    @ivar manager: See L{__init__}

    @type protocol: callable which returns L{SMTPManagedRelayer}
    @ivar protocol: A callable which returns a managed relayer for SMTP.  See
        L{SMTPManagedRelayer.__init__} for parameters to the callable.

    @type pArgs: 1-L{tuple} of (0) L{bytes} or 2-L{tuple} of
        (0) L{bytes}, (1), L{int}
    @ivar pArgs: Positional arguments for L{SMTPClient.__init__}

    @type pKwArgs: L{dict}
    @ivar pKwArgs: Keyword arguments for L{SMTPClient.__init__}
    """
    protocol: 'Type[protocol.Protocol]' = SMTPManagedRelayer

    def __init__(self, messages, manager, *args, **kw):
        """
        @type messages: L{list} of L{bytes}
        @param messages: The base filenames of messages to be relayed.

        @type manager: L{_AttemptManager}
        @param manager: An attempt manager.

        @type args: 1-L{tuple} of (0) L{bytes} or 2-L{tuple} of
            (0) L{bytes}, (1), L{int}
        @param args: Positional arguments for L{SMTPClient.__init__}

        @type kw: L{dict}
        @param kw: Keyword arguments for L{SMTPClient.__init__}
        """
        self.messages = messages
        self.manager = manager
        self.pArgs = args
        self.pKwArgs = kw

    def buildProtocol(self, addr):
        """
        Create an L{SMTPManagedRelayer}.

        @type addr: L{IAddress <twisted.internet.interfaces.IAddress>} provider
        @param addr: The address of the SMTP server.

        @rtype: L{SMTPManagedRelayer}
        @return: A managed relayer for SMTP.
        """
        protocol = self.protocol(self.messages, self.manager, *self.pArgs, **self.pKwArgs)
        protocol.factory = self
        return protocol

    def clientConnectionFailed(self, connector, reason):
        """
        Notify the attempt manager that a connection could not be established.

        @type connector: L{IConnector <twisted.internet.interfaces.IConnector>}
            provider
        @param connector: A connector.

        @type reason: L{Failure}
        @param reason: The reason the connection attempt failed.
        """
        self.manager.notifyNoConnection(self)
        self.manager.notifyDone(self)