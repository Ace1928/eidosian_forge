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
def _writeHeaders(self, transport, TEorCL):
    hosts = self.headers.getRawHeaders(b'host', ())
    if len(hosts) != 1:
        raise BadHeaders('Exactly one Host header required')
    requestLines = []
    requestLines.append(b' '.join([_ensureValidMethod(self.method), _ensureValidURI(self.uri), b'HTTP/1.1\r\n']))
    if not self.persistent:
        requestLines.append(b'Connection: close\r\n')
    if TEorCL is not None:
        requestLines.append(TEorCL)
    for name, values in self.headers.getAllRawHeaders():
        requestLines.extend([name + b': ' + v + b'\r\n' for v in values])
    requestLines.append(b'\r\n')
    transport.writeSequence(requestLines)