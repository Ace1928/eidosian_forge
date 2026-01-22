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
class _AttemptManager:
    """
    A manager for an attempt to relay a set of messages to a mail exchange
    server.

    @ivar manager: See L{__init__}

    @type _completionDeferreds: L{list} of L{Deferred}
    @ivar _completionDeferreds: Deferreds which are to be notified when the
        attempt to relay is finished.
    """

    def __init__(self, manager, noisy=True, reactor=None):
        """
        @type manager: L{SmartHostSMTPRelayingManager}
        @param manager: A smart host.

        @type noisy: L{bool}
        @param noisy: A flag which determines whether informational log
            messages will be generated (L{True}) or not (L{False}).

        @type reactor: L{IReactorTime
            <twisted.internet.interfaces.IReactorTime>} provider
        @param reactor: A reactor which will be used to schedule delayed calls.
        """
        self.manager = manager
        self._completionDeferreds = []
        self.noisy = noisy
        if not reactor:
            from twisted.internet import reactor
        self.reactor = reactor

    def getCompletionDeferred(self):
        """
        Return a deferred which will fire when the attempt to relay is
        finished.

        @rtype: L{Deferred}
        @return: A deferred which will fire when the attempt to relay is
            finished.
        """
        self._completionDeferreds.append(Deferred())
        return self._completionDeferreds[-1]

    def _finish(self, relay, message):
        """
        Remove a message from the relay queue and from the smart host's list of
        messages being relayed.

        @type relay: L{SMTPManagedRelayerFactory}
        @param relay: The factory for the relayer which sent the message.

        @type message: L{bytes}
        @param message: The path of the file holding the message.
        """
        self.manager.managed[relay].remove(os.path.basename(message))
        self.manager.queue.done(message)

    def notifySuccess(self, relay, message):
        """
        Remove a message from the relay queue after it has been successfully
        sent.

        @type relay: L{SMTPManagedRelayerFactory}
        @param relay: The factory for the relayer which sent the message.

        @type message: L{bytes}
        @param message: The path of the file holding the message.
        """
        if self.noisy:
            log.msg('success sending %s, removing from queue' % message)
        self._finish(relay, message)

    def notifyFailure(self, relay, message):
        """
        Generate a bounce message for a message which cannot be relayed.

        @type relay: L{SMTPManagedRelayerFactory}
        @param relay: The factory for the relayer responsible for the message.

        @type message: L{bytes}
        @param message: The path of the file holding the message.
        """
        if self.noisy:
            log.msg('could not relay ' + message)
        message = os.path.basename(message)
        with self.manager.queue.getEnvelopeFile(message) as fp:
            from_, to = pickle.load(fp)
        from_, to, bounceMessage = bounce.generateBounce(open(self.manager.queue.getPath(message) + '-D'), from_, to)
        fp, outgoingMessage = self.manager.queue.createNewMessage()
        with fp:
            pickle.dump([from_, to], fp)
        for line in bounceMessage.splitlines():
            outgoingMessage.lineReceived(line)
        outgoingMessage.eomReceived()
        self._finish(relay, self.manager.queue.getPath(message))

    def notifyDone(self, relay):
        """
        When the connection is lost or cannot be established, prepare to
        resend unsent messages and fire all deferred which are waiting for
        the completion of the attempt to relay.

        @type relay: L{SMTPManagedRelayerFactory}
        @param relay: The factory for the relayer for the connection.
        """
        for message in self.manager.managed.get(relay, ()):
            if self.noisy:
                log.msg('Setting ' + message + ' waiting')
            self.manager.queue.setWaiting(message)
        try:
            del self.manager.managed[relay]
        except KeyError:
            pass
        notifications = self._completionDeferreds
        self._completionDeferreds = None
        for d in notifications:
            d.callback(None)

    def notifyNoConnection(self, relay):
        """
        When a connection to the mail exchange server cannot be established,
        prepare to resend messages later.

        @type relay: L{SMTPManagedRelayerFactory}
        @param relay: The factory for the relayer meant to use the connection.
        """
        try:
            msgs = self.manager.managed[relay]
        except KeyError:
            log.msg('notifyNoConnection passed unknown relay!')
            return
        if self.noisy:
            log.msg('Backing off on delivery of ' + str(msgs))

        def setWaiting(queue, messages):
            map(queue.setWaiting, messages)
        self.reactor.callLater(30, setWaiting, self.manager.queue, msgs)
        del self.manager.managed[relay]