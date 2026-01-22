import re
from zope.interface import implementer
from twisted.internet.defer import (
from twisted.internet.error import ConnectionDone
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.reflect import fullyQualifiedName
from twisted.web.http import (
from twisted.web.http_headers import Headers
from twisted.web.iweb import UNKNOWN_LENGTH, IClientRequest, IResponse
class LengthEnforcingConsumer:
    """
    An L{IConsumer} proxy which enforces an exact length requirement on the
    total data written to it.

    @ivar _length: The number of bytes remaining to be written.

    @ivar _producer: The L{IBodyProducer} which is writing to this
        consumer.

    @ivar _consumer: The consumer to which at most C{_length} bytes will be
        forwarded.

    @ivar _finished: A L{Deferred} which will be fired with a L{Failure} if too
        many bytes are written to this consumer.
    """

    def __init__(self, producer, consumer, finished):
        self._length = producer.length
        self._producer = producer
        self._consumer = consumer
        self._finished = finished

    def _allowNoMoreWrites(self):
        """
        Indicate that no additional writes are allowed.  Attempts to write
        after calling this method will be met with an exception.
        """
        self._finished = None

    def write(self, bytes):
        """
        Write C{bytes} to the underlying consumer unless
        C{_noMoreWritesExpected} has been called or there are/have been too
        many bytes.
        """
        if self._finished is None:
            self._producer.stopProducing()
            raise ExcessWrite()
        if len(bytes) <= self._length:
            self._length -= len(bytes)
            self._consumer.write(bytes)
        else:
            _callAppFunction(self._producer.stopProducing)
            self._finished.errback(WrongBodyLength('too many bytes written'))
            self._allowNoMoreWrites()

    def _noMoreWritesExpected(self):
        """
        Called to indicate no more bytes will be written to this consumer.
        Check to see that the correct number have been written.

        @raise WrongBodyLength: If not enough bytes have been written.
        """
        if self._finished is not None:
            self._allowNoMoreWrites()
            if self._length:
                raise WrongBodyLength('too few bytes written')