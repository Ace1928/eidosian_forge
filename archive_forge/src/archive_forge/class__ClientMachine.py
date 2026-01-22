from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
class _ClientMachine:
    """
    State machine for maintaining a single outgoing connection to an endpoint.

    @ivar _awaitingConnected: notifications to make when connection
        succeeds, fails, or is cancelled
    @type _awaitingConnected: list of (Deferred, count) tuples

    @see: L{ClientService}
    """
    _machine = MethodicalMachine()

    def __init__(self, endpoint, factory, retryPolicy, clock, prepareConnection, log):
        """
        @see: L{ClientService.__init__}

        @param log: The logger for the L{ClientService} instance this state
            machine is associated to.
        @type log: L{Logger}
        """
        self._endpoint = endpoint
        self._failedAttempts = 0
        self._stopped = False
        self._factory = factory
        self._timeoutForAttempt = retryPolicy
        self._clock = clock
        self._prepareConnection = prepareConnection
        self._connectionInProgress = succeed(None)
        self._awaitingConnected = []
        self._stopWaiters = []
        self._log = log

    @_machine.state(initial=True)
    def _init(self):
        """
        The service has not been started.
        """

    @_machine.state()
    def _connecting(self):
        """
        The service has started connecting.
        """

    @_machine.state()
    def _waiting(self):
        """
        The service is waiting for the reconnection period
        before reconnecting.
        """

    @_machine.state()
    def _connected(self):
        """
        The service is connected.
        """

    @_machine.state()
    def _disconnecting(self):
        """
        The service is disconnecting after being asked to shutdown.
        """

    @_machine.state()
    def _restarting(self):
        """
        The service is disconnecting and has been asked to restart.
        """

    @_machine.state()
    def _stopped(self):
        """
        The service has been stopped and is disconnected.
        """

    @_machine.input()
    def start(self):
        """
        Start this L{ClientService}, initiating the connection retry loop.
        """

    @_machine.output()
    def _connect(self):
        """
        Start a connection attempt.
        """
        factoryProxy = _DisconnectFactory(self._factory, lambda _: self._clientDisconnected())
        self._connectionInProgress = self._endpoint.connect(factoryProxy).addCallback(self._runPrepareConnection).addCallback(self._connectionMade).addErrback(self._connectionFailed)

    def _runPrepareConnection(self, protocol):
        """
        Run any C{prepareConnection} callback with the connected protocol,
        ignoring its return value but propagating any failure.

        @param protocol: The protocol of the connection.
        @type protocol: L{IProtocol}

        @return: Either:

            - A L{Deferred} that succeeds with the protocol when the
              C{prepareConnection} callback has executed successfully.

            - A L{Deferred} that fails when the C{prepareConnection} callback
              throws or returns a failed L{Deferred}.

            - The protocol, when no C{prepareConnection} callback is defined.
        """
        if self._prepareConnection:
            return maybeDeferred(self._prepareConnection, protocol).addCallback(lambda _: protocol)
        return protocol

    @_machine.output()
    def _resetFailedAttempts(self):
        """
        Reset the number of failed attempts.
        """
        self._failedAttempts = 0

    @_machine.input()
    def stop(self):
        """
        Stop trying to connect and disconnect any current connection.

        @return: a L{Deferred} that fires when all outstanding connections are
            closed and all in-progress connection attempts halted.
        """

    @_machine.output()
    def _waitForStop(self):
        """
        Return a deferred that will fire when the service has finished
        disconnecting.

        @return: L{Deferred} that fires when the service has finished
            disconnecting.
        """
        self._stopWaiters.append(Deferred())
        return self._stopWaiters[-1]

    @_machine.output()
    def _stopConnecting(self):
        """
        Stop pending connection attempt.
        """
        self._connectionInProgress.cancel()

    @_machine.output()
    def _stopRetrying(self):
        """
        Stop pending attempt to reconnect.
        """
        self._retryCall.cancel()
        del self._retryCall

    @_machine.output()
    def _disconnect(self):
        """
        Disconnect the current connection.
        """
        self._currentConnection.transport.loseConnection()

    @_machine.input()
    def _connectionMade(self, protocol):
        """
        A connection has been made.

        @param protocol: The protocol of the connection.
        @type protocol: L{IProtocol}
        """

    @_machine.output()
    def _notifyWaiters(self, protocol):
        """
        Notify all pending requests for a connection that a connection has been
        made.

        @param protocol: The protocol of the connection.
        @type protocol: L{IProtocol}
        """
        self._failedAttempts = 0
        self._currentConnection = protocol._protocol
        self._unawait(self._currentConnection)

    @_machine.input()
    def _connectionFailed(self, f):
        """
        The current connection attempt failed.
        """

    @_machine.output()
    def _wait(self):
        """
        Schedule a retry attempt.
        """
        self._doWait()

    @_machine.output()
    def _ignoreAndWait(self, f):
        """
        Schedule a retry attempt, and ignore the Failure passed in.
        """
        return self._doWait()

    def _doWait(self):
        self._failedAttempts += 1
        delay = self._timeoutForAttempt(self._failedAttempts)
        self._log.info('Scheduling retry {attempt} to connect {endpoint} in {delay} seconds.', attempt=self._failedAttempts, endpoint=self._endpoint, delay=delay)
        self._retryCall = self._clock.callLater(delay, self._reconnect)

    @_machine.input()
    def _reconnect(self):
        """
        The wait between connection attempts is done.
        """

    @_machine.input()
    def _clientDisconnected(self):
        """
        The current connection has been disconnected.
        """

    @_machine.output()
    def _forgetConnection(self):
        """
        Forget the current connection.
        """
        del self._currentConnection

    @_machine.output()
    def _cancelConnectWaiters(self):
        """
        Notify all pending requests for a connection that no more connections
        are expected.
        """
        self._unawait(Failure(CancelledError()))

    @_machine.output()
    def _ignoreAndCancelConnectWaiters(self, f):
        """
        Notify all pending requests for a connection that no more connections
        are expected, after ignoring the Failure passed in.
        """
        self._unawait(Failure(CancelledError()))

    @_machine.output()
    def _finishStopping(self):
        """
        Notify all deferreds waiting on the service stopping.
        """
        self._doFinishStopping()

    @_machine.output()
    def _ignoreAndFinishStopping(self, f):
        """
        Notify all deferreds waiting on the service stopping, and ignore the
        Failure passed in.
        """
        self._doFinishStopping()

    def _doFinishStopping(self):
        self._stopWaiters, waiting = ([], self._stopWaiters)
        for w in waiting:
            w.callback(None)

    @_machine.input()
    def whenConnected(self, failAfterFailures=None):
        """
        Retrieve the currently-connected L{Protocol}, or the next one to
        connect.

        @param failAfterFailures: number of connection failures after which
            the Deferred will deliver a Failure (None means the Deferred will
            only fail if/when the service is stopped).  Set this to 1 to make
            the very first connection failure signal an error.  Use 2 to
            allow one failure but signal an error if the subsequent retry
            then fails.
        @type failAfterFailures: L{int} or None

        @return: a Deferred that fires with a protocol produced by the
            factory passed to C{__init__}
        @rtype: L{Deferred} that may:

            - fire with L{IProtocol}

            - fail with L{CancelledError} when the service is stopped

            - fail with e.g.
              L{DNSLookupError<twisted.internet.error.DNSLookupError>} or
              L{ConnectionRefusedError<twisted.internet.error.ConnectionRefusedError>}
              when the number of consecutive failed connection attempts
              equals the value of "failAfterFailures"
        """

    @_machine.output()
    def _currentConnection(self, failAfterFailures=None):
        """
        Return the currently connected protocol.

        @return: L{Deferred} that is fired with currently connected protocol.
        """
        return succeed(self._currentConnection)

    @_machine.output()
    def _noConnection(self, failAfterFailures=None):
        """
        Notify the caller that no connection is expected.

        @return: L{Deferred} that is fired with L{CancelledError}.
        """
        return fail(CancelledError())

    @_machine.output()
    def _awaitingConnection(self, failAfterFailures=None):
        """
        Return a deferred that will fire with the next connected protocol.

        @return: L{Deferred} that will fire with the next connected protocol.
        """
        result = Deferred()
        self._awaitingConnected.append((result, failAfterFailures))
        return result

    @_machine.output()
    def _deferredSucceededWithNone(self):
        """
        Return a deferred that has already fired with L{None}.

        @return: A L{Deferred} that has already fired with L{None}.
        """
        return succeed(None)

    def _unawait(self, value):
        """
        Fire all outstanding L{ClientService.whenConnected} L{Deferred}s.

        @param value: the value to fire the L{Deferred}s with.
        """
        self._awaitingConnected, waiting = ([], self._awaitingConnected)
        for w, remaining in waiting:
            w.callback(value)

    @_machine.output()
    def _deliverConnectionFailure(self, f):
        """
        Deliver connection failures to any L{ClientService.whenConnected}
        L{Deferred}s that have met their failAfterFailures threshold.

        @param f: the Failure to fire the L{Deferred}s with.
        """
        ready = []
        notReady = []
        for w, remaining in self._awaitingConnected:
            if remaining is None:
                notReady.append((w, remaining))
            elif remaining <= 1:
                ready.append(w)
            else:
                notReady.append((w, remaining - 1))
        self._awaitingConnected = notReady
        for w in ready:
            w.callback(f)
    _init.upon(start, enter=_connecting, outputs=[_connect])
    _init.upon(stop, enter=_stopped, outputs=[_deferredSucceededWithNone], collector=_firstResult)
    _connecting.upon(start, enter=_connecting, outputs=[])
    _connecting.upon(stop, enter=_disconnecting, outputs=[_waitForStop, _stopConnecting], collector=_firstResult)
    _connecting.upon(_connectionMade, enter=_connected, outputs=[_notifyWaiters])
    _connecting.upon(_connectionFailed, enter=_waiting, outputs=[_ignoreAndWait, _deliverConnectionFailure])
    _waiting.upon(start, enter=_waiting, outputs=[])
    _waiting.upon(stop, enter=_stopped, outputs=[_waitForStop, _cancelConnectWaiters, _stopRetrying, _finishStopping], collector=_firstResult)
    _waiting.upon(_reconnect, enter=_connecting, outputs=[_connect])
    _connected.upon(start, enter=_connected, outputs=[])
    _connected.upon(stop, enter=_disconnecting, outputs=[_waitForStop, _disconnect], collector=_firstResult)
    _connected.upon(_clientDisconnected, enter=_waiting, outputs=[_forgetConnection, _wait])
    _disconnecting.upon(start, enter=_restarting, outputs=[_resetFailedAttempts])
    _disconnecting.upon(stop, enter=_disconnecting, outputs=[_waitForStop], collector=_firstResult)
    _disconnecting.upon(_clientDisconnected, enter=_stopped, outputs=[_cancelConnectWaiters, _finishStopping, _forgetConnection])
    _disconnecting.upon(_connectionFailed, enter=_stopped, outputs=[_ignoreAndCancelConnectWaiters, _ignoreAndFinishStopping])
    _restarting.upon(start, enter=_restarting, outputs=[])
    _restarting.upon(stop, enter=_disconnecting, outputs=[_waitForStop], collector=_firstResult)
    _restarting.upon(_clientDisconnected, enter=_connecting, outputs=[_finishStopping, _connect])
    _stopped.upon(start, enter=_connecting, outputs=[_connect])
    _stopped.upon(stop, enter=_stopped, outputs=[_deferredSucceededWithNone], collector=_firstResult)
    _init.upon(whenConnected, enter=_init, outputs=[_awaitingConnection], collector=_firstResult)
    _connecting.upon(whenConnected, enter=_connecting, outputs=[_awaitingConnection], collector=_firstResult)
    _waiting.upon(whenConnected, enter=_waiting, outputs=[_awaitingConnection], collector=_firstResult)
    _connected.upon(whenConnected, enter=_connected, outputs=[_currentConnection], collector=_firstResult)
    _disconnecting.upon(whenConnected, enter=_disconnecting, outputs=[_awaitingConnection], collector=_firstResult)
    _restarting.upon(whenConnected, enter=_restarting, outputs=[_awaitingConnection], collector=_firstResult)
    _stopped.upon(whenConnected, enter=_stopped, outputs=[_noConnection], collector=_firstResult)