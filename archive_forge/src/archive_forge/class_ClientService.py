from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
class ClientService(service.Service):
    """
    A L{ClientService} maintains a single outgoing connection to a client
    endpoint, reconnecting after a configurable timeout when a connection
    fails, either before or after connecting.

    @since: 16.1.0
    """
    _log = Logger()

    def __init__(self, endpoint, factory, retryPolicy=None, clock=None, prepareConnection=None):
        """
        @param endpoint: A L{stream client endpoint
            <interfaces.IStreamClientEndpoint>} provider which will be used to
            connect when the service starts.

        @param factory: A L{protocol factory <interfaces.IProtocolFactory>}
            which will be used to create clients for the endpoint.

        @param retryPolicy: A policy configuring how long L{ClientService} will
            wait between attempts to connect to C{endpoint}.
        @type retryPolicy: callable taking (the number of failed connection
            attempts made in a row (L{int})) and returning the number of
            seconds to wait before making another attempt.

        @param clock: The clock used to schedule reconnection.  It's mainly
            useful to be parametrized in tests.  If the factory is serialized,
            this attribute will not be serialized, and the default value (the
            reactor) will be restored when deserialized.
        @type clock: L{IReactorTime}

        @param prepareConnection: A single argument L{callable} that may return
            a L{Deferred}. It will be called once with the L{protocol
            <interfaces.IProtocol>} each time a new connection is made.  It may
            call methods on the protocol to prepare it for use (e.g.
            authenticate) or validate it (check its health).

            The C{prepareConnection} callable may raise an exception or return
            a L{Deferred} which fails to reject the connection.  A rejected
            connection is not used to fire an L{Deferred} returned by
            L{whenConnected}.  Instead, L{ClientService} handles the failure
            and continues as if the connection attempt were a failure
            (incrementing the counter passed to C{retryPolicy}).

            L{Deferred}s returned by L{whenConnected} will not fire until
            any L{Deferred} returned by the C{prepareConnection} callable
            fire. Otherwise its successful return value is consumed, but
            ignored.

            Present Since Twisted 18.7.0

        @type prepareConnection: L{callable}

        """
        clock = _maybeGlobalReactor(clock)
        retryPolicy = _defaultPolicy if retryPolicy is None else retryPolicy
        self._machine = _ClientMachine(endpoint, factory, retryPolicy, clock, prepareConnection=prepareConnection, log=self._log)

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
        return self._machine.whenConnected(failAfterFailures)

    def startService(self):
        """
        Start this L{ClientService}, initiating the connection retry loop.
        """
        if self.running:
            self._log.warn('Duplicate ClientService.startService {log_source}')
            return
        super().startService()
        self._machine.start()

    def stopService(self):
        """
        Stop attempting to reconnect and close any existing connections.

        @return: a L{Deferred} that fires when all outstanding connections are
            closed and all in-progress connection attempts halted.
        """
        super().stopService()
        return self._machine.stop()