from socket import AF_INET, AF_INET6, inet_pton
from typing import Iterable, List, Optional
from zope.interface import implementer
from twisted.internet import interfaces, main
from twisted.python import failure, reflect
from twisted.python.compat import lazyByteSlice
class _ConsumerMixin:
    """
    L{IConsumer} implementations can mix this in to get C{registerProducer} and
    C{unregisterProducer} methods which take care of keeping track of a
    producer's state.

    Subclasses must provide three attributes which L{_ConsumerMixin} will read
    but not write:

      - connected: A C{bool} which is C{True} as long as the consumer has
        someplace to send bytes (for example, a TCP connection), and then
        C{False} when it no longer does.

      - disconnecting: A C{bool} which is C{False} until something like
        L{ITransport.loseConnection} is called, indicating that the send buffer
        should be flushed and the connection lost afterwards.  Afterwards,
        C{True}.

      - disconnected: A C{bool} which is C{False} until the consumer no longer
        has a place to send bytes, then C{True}.

    Subclasses must also override the C{startWriting} method.

    @ivar producer: L{None} if no producer is registered, otherwise the
        registered producer.

    @ivar producerPaused: A flag indicating whether the producer is currently
        paused.
    @type producerPaused: L{bool}

    @ivar streamingProducer: A flag indicating whether the producer was
        registered as a streaming (ie push) producer or not (ie a pull
        producer).  This will determine whether the consumer may ever need to
        pause and resume it, or if it can merely call C{resumeProducing} on it
        when buffer space is available.
    @ivar streamingProducer: C{bool} or C{int}

    """
    producer = None
    producerPaused = False
    streamingProducer = False

    def startWriting(self):
        """
        Override in a subclass to cause the reactor to monitor this selectable
        for write events.  This will be called once in C{unregisterProducer} if
        C{loseConnection} has previously been called, so that the connection can
        actually close.
        """
        raise NotImplementedError('%r did not implement startWriting')

    def registerProducer(self, producer, streaming):
        """
        Register to receive data from a producer.

        This sets this selectable to be a consumer for a producer.  When this
        selectable runs out of data on a write() call, it will ask the producer
        to resumeProducing(). When the FileDescriptor's internal data buffer is
        filled, it will ask the producer to pauseProducing(). If the connection
        is lost, FileDescriptor calls producer's stopProducing() method.

        If streaming is true, the producer should provide the IPushProducer
        interface. Otherwise, it is assumed that producer provides the
        IPullProducer interface. In this case, the producer won't be asked to
        pauseProducing(), but it has to be careful to write() data only when its
        resumeProducing() method is called.
        """
        if self.producer is not None:
            raise RuntimeError('Cannot register producer %s, because producer %s was never unregistered.' % (producer, self.producer))
        if self.disconnected:
            producer.stopProducing()
        else:
            self.producer = producer
            self.streamingProducer = streaming
            if not streaming:
                producer.resumeProducing()

    def unregisterProducer(self):
        """
        Stop consuming data from a producer, without disconnecting.
        """
        self.producer = None
        if self.connected and self.disconnecting:
            self.startWriting()