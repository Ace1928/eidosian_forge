import os
import tempfile
from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.mail import smtp
from twisted.mail.interfaces import IAlias
from twisted.python import failure, log
@implementer(smtp.IMessage)
class MessageWrapper:
    """
    A message receiver which delivers a message to a child process.

    @type completionTimeout: L{int} or L{float}
    @ivar completionTimeout: The number of seconds to wait for the child
        process to exit before reporting the delivery as a failure.

    @type _timeoutCallID: L{None} or
        L{IDelayedCall <twisted.internet.interfaces.IDelayedCall>} provider
    @ivar _timeoutCallID: The call used to time out delivery, started when the
        connection to the child process is closed.

    @type done: L{bool}
    @ivar done: A flag indicating whether the child process has exited
        (C{True}) or not (C{False}).

    @type reactor: L{IReactorTime <twisted.internet.interfaces.IReactorTime>}
        provider
    @ivar reactor: A reactor which will be used to schedule timeouts.

    @ivar protocol: See L{__init__}.

    @type processName: L{bytes} or L{None}
    @ivar processName: The process name.

    @type completion: L{Deferred <defer.Deferred>}
    @ivar completion: The deferred which will be triggered by the protocol
        when the child process exits.
    """
    done = False
    completionTimeout = 60
    _timeoutCallID = None
    reactor = reactor

    def __init__(self, protocol, process=None, reactor=None):
        """
        @type protocol: L{ProcessAliasProtocol}
        @param protocol: The protocol associated with the child process.

        @type process: L{bytes} or L{None}
        @param process: The process name.

        @type reactor: L{None} or L{IReactorTime
            <twisted.internet.interfaces.IReactorTime>} provider
        @param reactor: A reactor which will be used to schedule timeouts.
        """
        self.processName = process
        self.protocol = protocol
        self.completion = defer.Deferred()
        self.protocol.onEnd = self.completion
        self.completion.addBoth(self._processEnded)
        if reactor is not None:
            self.reactor = reactor

    def _processEnded(self, result):
        """
        Record process termination and cancel the timeout call if it is active.

        @type result: L{Failure <failure.Failure>}
        @param result: The reason the child process terminated.

        @rtype: L{None} or L{Failure <failure.Failure>}
        @return: None, if the process end is expected, or the reason the child
            process terminated, if the process end is unexpected.
        """
        self.done = True
        if self._timeoutCallID is not None:
            self._timeoutCallID.cancel()
            self._timeoutCallID = None
        else:
            return result

    def lineReceived(self, line):
        """
        Write a received line to the child process.

        @type line: L{bytes}
        @param line: A received line of the message.
        """
        if self.done:
            return
        self.protocol.transport.write(line + '\n')

    def eomReceived(self):
        """
        Disconnect from the child process and set up a timeout to wait for it
        to exit.

        @rtype: L{Deferred <defer.Deferred>}
        @return: A deferred which will be called back when the child process
            exits.
        """
        if not self.done:
            self.protocol.transport.loseConnection()
            self._timeoutCallID = self.reactor.callLater(self.completionTimeout, self._completionCancel)
        return self.completion

    def _completionCancel(self):
        """
        Handle the expiration of the timeout for the child process to exit by
        terminating the child process forcefully and issuing a failure to the
        L{completion} deferred.
        """
        self._timeoutCallID = None
        self.protocol.transport.signalProcess('KILL')
        exc = ProcessAliasTimeout(f'No answer after {self.completionTimeout} seconds')
        self.protocol.onEnd = None
        self.completion.errback(failure.Failure(exc))

    def connectionLost(self):
        """
        Ignore notification of lost connection.
        """

    def __str__(self) -> str:
        """
        Build a string representation of this L{MessageWrapper} instance.

        @rtype: L{bytes}
        @return: A string containing the name of the process.
        """
        return f'<ProcessWrapper {self.processName}>'