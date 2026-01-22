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
def _finishResponse_WAITING(self, rest):
    if self._state == 'WAITING':
        self._state = 'QUIESCENT'
    else:
        self._state = 'TRANSMITTING_AFTER_RECEIVING_RESPONSE'
        self._responseDeferred.chainDeferred(self._finishedRequest)
    if self._parser is None:
        return
    reason = ConnectionDone('synthetic!')
    connHeaders = self._parser.connHeaders.getRawHeaders(b'connection', ())
    if b'close' in connHeaders or self._state != 'QUIESCENT' or (not self._currentRequest.persistent):
        self._giveUp(Failure(reason))
    else:
        self.transport.resumeProducing()
        try:
            self._quiescentCallback(self)
        except BaseException:
            self._log.failure('')
            self.transport.loseConnection()
        self._disconnectParser(reason)