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
class HTTPClientParser(HTTPParser):
    """
    An HTTP parser which only handles HTTP responses.

    @ivar request: The request with which the expected response is associated.
    @type request: L{Request}

    @ivar NO_BODY_CODES: A C{set} of response codes which B{MUST NOT} have a
        body.

    @ivar finisher: A callable to invoke when this response is fully parsed.

    @ivar _responseDeferred: A L{Deferred} which will be called back with the
        response when all headers in the response have been received.
        Thereafter, L{None}.

    @ivar _everReceivedData: C{True} if any bytes have been received.
    """
    NO_BODY_CODES = {NO_CONTENT, NOT_MODIFIED}
    _transferDecoders = {b'chunked': _ChunkedTransferDecoder}
    bodyDecoder = None
    _log = Logger()

    def __init__(self, request, finisher):
        self.request = request
        self.finisher = finisher
        self._responseDeferred = Deferred()
        self._everReceivedData = False

    def dataReceived(self, data):
        """
        Override so that we know if any response has been received.
        """
        self._everReceivedData = True
        HTTPParser.dataReceived(self, data)

    def parseVersion(self, strversion):
        """
        Parse version strings of the form Protocol '/' Major '.' Minor. E.g.
        b'HTTP/1.1'.  Returns (protocol, major, minor).  Will raise ValueError
        on bad syntax.
        """
        try:
            proto, strnumber = strversion.split(b'/')
            major, minor = strnumber.split(b'.')
            major, minor = (int(major), int(minor))
        except ValueError as e:
            raise BadResponseVersion(str(e), strversion)
        if major < 0 or minor < 0:
            raise BadResponseVersion('version may not be negative', strversion)
        return (proto, major, minor)

    def statusReceived(self, status):
        """
        Parse the status line into its components and create a response object
        to keep track of this response's state.
        """
        parts = status.split(b' ', 2)
        if len(parts) == 2:
            version, codeBytes = parts
            phrase = b''
        elif len(parts) == 3:
            version, codeBytes, phrase = parts
        else:
            raise ParseError('wrong number of parts', status)
        try:
            statusCode = int(codeBytes)
        except ValueError:
            raise ParseError('non-integer status code', status)
        self.response = Response._construct(self.parseVersion(version), statusCode, phrase, self.headers, self.transport, self.request)

    def _finished(self, rest):
        """
        Called to indicate that an entire response has been received.  No more
        bytes will be interpreted by this L{HTTPClientParser}.  Extra bytes are
        passed up and the state of this L{HTTPClientParser} is set to I{DONE}.

        @param rest: A C{bytes} giving any extra bytes delivered to this
            L{HTTPClientParser} which are not part of the response being
            parsed.
        """
        self.state = DONE
        self.finisher(rest)

    def isConnectionControlHeader(self, name):
        """
        Content-Length in the response to a HEAD request is an entity header,
        not a connection control header.
        """
        if self.request.method == b'HEAD' and name == b'content-length':
            return False
        return HTTPParser.isConnectionControlHeader(self, name)

    def allHeadersReceived(self):
        """
        Figure out how long the response body is going to be by examining
        headers and stuff.
        """
        if 100 <= self.response.code < 200:
            self._log.info('Ignoring unexpected {code} response', code=self.response.code)
            self.connectionMade()
            del self.response
            return
        if self.response.code in self.NO_BODY_CODES or self.request.method == b'HEAD':
            self.response.length = 0
            self._finished(self.clearLineBuffer())
            self.response._bodyDataFinished()
        else:
            transferEncodingHeaders = self.connHeaders.getRawHeaders(b'transfer-encoding')
            if transferEncodingHeaders:
                transferDecoder = self._transferDecoders[transferEncodingHeaders[0].lower()]
            else:
                contentLengthHeaders = self.connHeaders.getRawHeaders(b'content-length')
                if contentLengthHeaders is None:
                    contentLength = None
                elif len(contentLengthHeaders) == 1:
                    contentLength = int(contentLengthHeaders[0])
                    self.response.length = contentLength
                else:
                    raise ValueError('Too many Content-Length headers; response is invalid')
                if contentLength == 0:
                    self._finished(self.clearLineBuffer())
                    transferDecoder = None
                else:
                    transferDecoder = lambda x, y: _IdentityTransferDecoder(contentLength, x, y)
            if transferDecoder is None:
                self.response._bodyDataFinished()
            else:
                self.transport.pauseProducing()
                self.switchToBodyMode(transferDecoder(self.response._bodyDataReceived, self._finished))
        self._responseDeferred.callback(self.response)
        del self._responseDeferred

    def connectionLost(self, reason):
        if self.bodyDecoder is not None:
            try:
                try:
                    self.bodyDecoder.noMoreData()
                except PotentialDataLoss:
                    self.response._bodyDataFinished(Failure())
                except _DataLoss:
                    self.response._bodyDataFinished(Failure(ResponseFailed([reason, Failure()], self.response)))
                else:
                    self.response._bodyDataFinished()
            except BaseException:
                self._log.failure('')
        elif self.state != DONE:
            if self._everReceivedData:
                exceptionClass = ResponseFailed
            else:
                exceptionClass = ResponseNeverReceived
            self._responseDeferred.errback(Failure(exceptionClass([reason])))
            del self._responseDeferred