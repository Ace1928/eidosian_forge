import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
@implementer(interfaces.IProducer)
class FileSender:
    """
    A producer that sends the contents of a file to a consumer.

    This is a helper for protocols that, at some point, will take a
    file-like object, read its contents, and write them out to the network,
    optionally performing some transformation on the bytes in between.
    """
    CHUNK_SIZE = 2 ** 14
    lastSent = ''
    deferred = None

    def beginFileTransfer(self, file, consumer, transform=None):
        """
        Begin transferring a file

        @type file: Any file-like object
        @param file: The file object to read data from

        @type consumer: Any implementor of IConsumer
        @param consumer: The object to write data to

        @param transform: A callable taking one string argument and returning
        the same.  All bytes read from the file are passed through this before
        being written to the consumer.

        @rtype: C{Deferred}
        @return: A deferred whose callback will be invoked when the file has
        been completely written to the consumer. The last byte written to the
        consumer is passed to the callback.
        """
        self.file = file
        self.consumer = consumer
        self.transform = transform
        self.deferred = deferred = defer.Deferred()
        self.consumer.registerProducer(self, False)
        return deferred

    def resumeProducing(self):
        chunk = ''
        if self.file:
            chunk = self.file.read(self.CHUNK_SIZE)
        if not chunk:
            self.file = None
            self.consumer.unregisterProducer()
            if self.deferred:
                self.deferred.callback(self.lastSent)
                self.deferred = None
            return
        if self.transform:
            chunk = self.transform(chunk)
        self.consumer.write(chunk)
        self.lastSent = chunk[-1:]

    def pauseProducing(self):
        pass

    def stopProducing(self):
        if self.deferred:
            self.deferred.errback(Exception('Consumer asked us to stop producing'))
            self.deferred = None