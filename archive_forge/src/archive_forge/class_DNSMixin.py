from __future__ import annotations
import inspect
import random
import socket
import struct
from io import BytesIO
from itertools import chain
from typing import Optional, Sequence, SupportsInt, Union, overload
from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import CannotListenError
from twisted.python import failure, log, randbytes, util as tputil
from twisted.python.compat import cmp, comparable, nativeString
from twisted.names.error import (
class DNSMixin:
    """
    DNS protocol mixin shared by UDP and TCP implementations.

    @ivar _reactor: A L{IReactorTime} and L{IReactorUDP} provider which will
        be used to issue DNS queries and manage request timeouts.
    """
    id = None
    liveMessages = None

    def __init__(self, controller, reactor=None):
        self.controller = controller
        self.id = random.randrange(2 ** 10, 2 ** 15)
        if reactor is None:
            from twisted.internet import reactor
        self._reactor = reactor

    def pickID(self):
        """
        Return a unique ID for queries.
        """
        while True:
            id = randomSource()
            if id not in self.liveMessages:
                return id

    def callLater(self, period, func, *args):
        """
        Wrapper around reactor.callLater, mainly for test purpose.
        """
        return self._reactor.callLater(period, func, *args)

    def _query(self, queries, timeout, id, writeMessage):
        """
        Send out a message with the given queries.

        @type queries: L{list} of C{Query} instances
        @param queries: The queries to transmit

        @type timeout: L{int} or C{float}
        @param timeout: How long to wait before giving up

        @type id: L{int}
        @param id: Unique key for this request

        @type writeMessage: C{callable}
        @param writeMessage: One-parameter callback which writes the message

        @rtype: C{Deferred}
        @return: a C{Deferred} which will be fired with the result of the
            query, or errbacked with any errors that could happen (exceptions
            during writing of the query, timeout errors, ...).
        """
        m = Message(id, recDes=1)
        m.queries = queries
        try:
            writeMessage(m)
        except BaseException:
            return defer.fail()
        resultDeferred = defer.Deferred()
        cancelCall = self.callLater(timeout, self._clearFailed, resultDeferred, id)
        self.liveMessages[id] = (resultDeferred, cancelCall)
        return resultDeferred

    def _clearFailed(self, deferred, id):
        """
        Clean the Deferred after a timeout.
        """
        try:
            del self.liveMessages[id]
        except KeyError:
            pass
        deferred.errback(failure.Failure(DNSQueryTimeoutError(id)))