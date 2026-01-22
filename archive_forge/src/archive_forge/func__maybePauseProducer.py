from socket import AF_INET, AF_INET6, inet_pton
from typing import Iterable, List, Optional
from zope.interface import implementer
from twisted.internet import interfaces, main
from twisted.python import failure, reflect
from twisted.python.compat import lazyByteSlice
def _maybePauseProducer(self):
    """
        Possibly pause a producer, if there is one and the send buffer is full.
        """
    if self.producer is not None and self.streamingProducer:
        if self._isSendBufferFull():
            self.producerPaused = True
            self.producer.pauseProducing()