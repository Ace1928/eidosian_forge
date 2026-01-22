from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
class HTTPClientParserTests(TestCase):
    """
    Tests for L{HTTPClientParser} which is responsible for parsing HTTP
    response messages.
    """

    def test_parseVersion(self):
        """
        L{HTTPClientParser.parseVersion} parses a status line into its three
        components.
        """
        protocol = HTTPClientParser(None, None)
        self.assertEqual(protocol.parseVersion(b'CANDY/7.2'), (b'CANDY', 7, 2))

    def test_parseBadVersion(self):
        """
        L{HTTPClientParser.parseVersion} raises L{ValueError} when passed an
        unparsable version.
        """
        protocol = HTTPClientParser(None, None)
        e = BadResponseVersion
        f = protocol.parseVersion

        def checkParsing(s):
            exc = self.assertRaises(e, f, s)
            self.assertEqual(exc.data, s)
        checkParsing(b'foo')
        checkParsing(b'foo/bar/baz')
        checkParsing(b'foo/')
        checkParsing(b'foo/..')
        checkParsing(b'foo/a.b')
        checkParsing(b'foo/-1.-1')

    def test_responseStatusParsing(self):
        """
        L{HTTPClientParser.statusReceived} parses the version, code, and phrase
        from the status line and stores them on the response object.
        """
        request = Request(b'GET', b'/', _boringHeaders, None)
        protocol = HTTPClientParser(request, None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        self.assertEqual(protocol.response.version, (b'HTTP', 1, 1))
        self.assertEqual(protocol.response.code, 200)
        self.assertEqual(protocol.response.phrase, b'OK')

    def test_responseStatusWithoutPhrase(self):
        """
        L{HTTPClientParser.statusReceived} can parse a status line without a
        phrase (though such lines are a violation of RFC 7230, section 3.1.2;
        nevertheless some broken servers omit the phrase).
        """
        request = Request(b'GET', b'/', _boringHeaders, None)
        protocol = HTTPClientParser(request, None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200\r\n')
        self.assertEqual(protocol.response.version, (b'HTTP', 1, 1))
        self.assertEqual(protocol.response.code, 200)
        self.assertEqual(protocol.response.phrase, b'')

    def test_badResponseStatus(self):
        """
        L{HTTPClientParser.statusReceived} raises L{ParseError} if it is called
        with a status line which cannot be parsed.
        """
        protocol = HTTPClientParser(None, None)

        def checkParsing(s):
            exc = self.assertRaises(ParseError, protocol.statusReceived, s)
            self.assertEqual(exc.data, s)
        checkParsing(b'foo')
        checkParsing(b'HTTP/1.1 bar OK')

    def _noBodyTest(self, request, status, response):
        """
        Assert that L{HTTPClientParser} parses the given C{response} to
        C{request}, resulting in a response with no body and no extra bytes and
        leaving the transport in the producing state.

        @param request: A L{Request} instance which might have caused a server
            to return the given response.
        @param status: A string giving the status line of the response to be
            parsed.
        @param response: A string giving the response to be parsed.

        @return: A C{dict} of headers from the response.
        """
        header = {}
        finished = []
        body = []
        bodyDataFinished = []
        protocol = HTTPClientParser(request, finished.append)
        protocol.headerReceived = header.__setitem__
        transport = StringTransport()
        protocol.makeConnection(transport)
        protocol.dataReceived(status)
        protocol.response._bodyDataReceived = body.append
        protocol.response._bodyDataFinished = lambda: bodyDataFinished.append(True)
        protocol.dataReceived(response)
        self.assertEqual(transport.producerState, 'producing')
        self.assertEqual(protocol.state, DONE)
        self.assertEqual(body, [])
        self.assertEqual(finished, [b''])
        self.assertEqual(bodyDataFinished, [True])
        self.assertEqual(protocol.response.length, 0)
        return header

    def test_headResponse(self):
        """
        If the response is to a HEAD request, no body is expected, the body
        callback is not invoked, and the I{Content-Length} header is passed to
        the header callback.
        """
        request = Request(b'HEAD', b'/', _boringHeaders, None)
        status = b'HTTP/1.1 200 OK\r\n'
        response = b'Content-Length: 10\r\n\r\n'
        header = self._noBodyTest(request, status, response)
        self.assertEqual(header, {b'Content-Length': b'10'})

    def test_noContentResponse(self):
        """
        If the response code is I{NO CONTENT} (204), no body is expected and
        the body callback is not invoked.
        """
        request = Request(b'GET', b'/', _boringHeaders, None)
        status = b'HTTP/1.1 204 NO CONTENT\r\n'
        response = b'\r\n'
        self._noBodyTest(request, status, response)

    def test_notModifiedResponse(self):
        """
        If the response code is I{NOT MODIFIED} (304), no body is expected and
        the body callback is not invoked.
        """
        request = Request(b'GET', b'/', _boringHeaders, None)
        status = b'HTTP/1.1 304 NOT MODIFIED\r\n'
        response = b'\r\n'
        self._noBodyTest(request, status, response)

    def test_responseHeaders(self):
        """
        The response headers are added to the response object's C{headers}
        L{Headers} instance.
        """
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda rest: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        protocol.dataReceived(b'X-Foo: bar\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(protocol.connHeaders, Headers({}))
        self.assertEqual(protocol.response.headers, Headers({b'x-foo': [b'bar']}))
        self.assertIdentical(protocol.response.length, UNKNOWN_LENGTH)

    def test_responseHeadersMultiline(self):
        """
        The multi-line response headers are folded and added to the response
        object's C{headers} L{Headers} instance.
        """
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda rest: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        protocol.dataReceived(b'X-Multiline: a\r\n')
        protocol.dataReceived(b'    b\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(protocol.connHeaders, Headers({}))
        self.assertEqual(protocol.response.headers, Headers({b'x-multiline': [b'a    b']}))
        self.assertIdentical(protocol.response.length, UNKNOWN_LENGTH)

    def test_connectionHeaders(self):
        """
        The connection control headers are added to the parser's C{connHeaders}
        L{Headers} instance.
        """
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda rest: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        protocol.dataReceived(b'Content-Length: 123\r\n')
        protocol.dataReceived(b'Connection: close\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(protocol.response.headers, Headers({}))
        self.assertEqual(protocol.connHeaders, Headers({b'content-length': [b'123'], b'connection': [b'close']}))
        self.assertEqual(protocol.response.length, 123)

    def test_headResponseContentLengthEntityHeader(self):
        """
        If a HEAD request is made, the I{Content-Length} header in the response
        is added to the response headers, not the connection control headers.
        """
        protocol = HTTPClientParser(Request(b'HEAD', b'/', _boringHeaders, None), lambda rest: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        protocol.dataReceived(b'Content-Length: 123\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(protocol.response.headers, Headers({b'content-length': [b'123']}))
        self.assertEqual(protocol.connHeaders, Headers({}))
        self.assertEqual(protocol.response.length, 0)

    def test_contentLength(self):
        """
        If a response includes a body with a length given by the
        I{Content-Length} header, the bytes which make up the body are passed
        to the C{_bodyDataReceived} callback on the L{HTTPParser}.
        """
        finished = []
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
        transport = StringTransport()
        protocol.makeConnection(transport)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        body = []
        protocol.response._bodyDataReceived = body.append
        protocol.dataReceived(b'Content-Length: 10\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(transport.producerState, 'paused')
        self.assertEqual(protocol.state, BODY)
        protocol.dataReceived(b'x' * 6)
        self.assertEqual(body, [b'x' * 6])
        self.assertEqual(protocol.state, BODY)
        protocol.dataReceived(b'y' * 4)
        self.assertEqual(body, [b'x' * 6, b'y' * 4])
        self.assertEqual(protocol.state, DONE)
        self.assertEqual(finished, [b''])

    def test_zeroContentLength(self):
        """
        If a response includes a I{Content-Length} header indicating zero bytes
        in the response, L{Response.length} is set accordingly and no data is
        delivered to L{Response._bodyDataReceived}.
        """
        finished = []
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        body = []
        protocol.response._bodyDataReceived = body.append
        protocol.dataReceived(b'Content-Length: 0\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(protocol.state, DONE)
        self.assertEqual(body, [])
        self.assertEqual(finished, [b''])
        self.assertEqual(protocol.response.length, 0)

    def test_multipleContentLengthHeaders(self):
        """
        If a response includes multiple I{Content-Length} headers,
        L{HTTPClientParser.dataReceived} raises L{ValueError} to indicate that
        the response is invalid and the transport is now unusable.
        """
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), None)
        protocol.makeConnection(StringTransport())
        self.assertRaises(ValueError, protocol.dataReceived, b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n')

    def test_extraBytesPassedBack(self):
        """
        If extra bytes are received past the end of a response, they are passed
        to the finish callback.
        """
        finished = []
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        protocol.dataReceived(b'Content-Length: 0\r\n')
        protocol.dataReceived(b'\r\nHere is another thing!')
        self.assertEqual(protocol.state, DONE)
        self.assertEqual(finished, [b'Here is another thing!'])

    def test_extraBytesPassedBackHEAD(self):
        """
        If extra bytes are received past the end of the headers of a response
        to a HEAD request, they are passed to the finish callback.
        """
        finished = []
        protocol = HTTPClientParser(Request(b'HEAD', b'/', _boringHeaders, None), finished.append)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        protocol.dataReceived(b'Content-Length: 12\r\n')
        protocol.dataReceived(b'\r\nHere is another thing!')
        self.assertEqual(protocol.state, DONE)
        self.assertEqual(finished, [b'Here is another thing!'])

    def test_chunkedResponseBody(self):
        """
        If the response headers indicate the response body is encoded with the
        I{chunked} transfer encoding, the body is decoded according to that
        transfer encoding before being passed to L{Response._bodyDataReceived}.
        """
        finished = []
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        body = []
        protocol.response._bodyDataReceived = body.append
        protocol.dataReceived(b'Transfer-Encoding: chunked\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(body, [])
        self.assertIdentical(protocol.response.length, UNKNOWN_LENGTH)
        protocol.dataReceived(b'3\r\na')
        self.assertEqual(body, [b'a'])
        protocol.dataReceived(b'bc\r\n')
        self.assertEqual(body, [b'a', b'bc'])
        protocol.dataReceived(b'0\r\n\r\nextra')
        self.assertEqual(finished, [b'extra'])

    def test_unknownContentLength(self):
        """
        If a response does not include a I{Transfer-Encoding} or a
        I{Content-Length}, the end of response body is indicated by the
        connection being closed.
        """
        finished = []
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
        transport = StringTransport()
        protocol.makeConnection(transport)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        body = []
        protocol.response._bodyDataReceived = body.append
        protocol.dataReceived(b'\r\n')
        protocol.dataReceived(b'foo')
        protocol.dataReceived(b'bar')
        self.assertEqual(body, [b'foo', b'bar'])
        protocol.connectionLost(ConnectionDone('simulated end of connection'))
        self.assertEqual(finished, [b''])

    def test_contentLengthAndTransferEncoding(self):
        """
        According to RFC 2616, section 4.4, point 3, if I{Content-Length} and
        I{Transfer-Encoding: chunked} are present, I{Content-Length} MUST be
        ignored
        """
        finished = []
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
        transport = StringTransport()
        protocol.makeConnection(transport)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        body = []
        protocol.response._bodyDataReceived = body.append
        protocol.dataReceived(b'Content-Length: 102\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n')
        self.assertEqual(body, [b'abc'])
        self.assertEqual(finished, [b''])

    def test_connectionLostBeforeBody(self):
        """
        If L{HTTPClientParser.connectionLost} is called before the headers are
        finished, the C{_responseDeferred} is fired with the L{Failure} passed
        to C{connectionLost}.
        """
        transport = StringTransport()
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), None)
        protocol.makeConnection(transport)
        responseDeferred = protocol._responseDeferred
        protocol.connectionLost(Failure(ArbitraryException()))
        return assertResponseFailed(self, responseDeferred, [ArbitraryException])

    def test_connectionLostWithError(self):
        """
        If one of the L{Response} methods called by
        L{HTTPClientParser.connectionLost} raises an exception, the exception
        is logged and not re-raised.
        """
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        transport = StringTransport()
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), None)
        protocol.makeConnection(transport)
        response = []
        protocol._responseDeferred.addCallback(response.append)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n')
        response = response[0]

        def fakeBodyDataFinished(err=None):
            raise ArbitraryException()
        response._bodyDataFinished = fakeBodyDataFinished
        protocol.connectionLost(None)
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        f = event['log_failure']
        self.assertIsInstance(f.value, ArbitraryException)
        self.flushLoggedErrors(ArbitraryException)

    def test_noResponseAtAll(self):
        """
        If no response at all was received and the connection is lost, the
        resulting error is L{ResponseNeverReceived}.
        """
        protocol = HTTPClientParser(Request(b'HEAD', b'/', _boringHeaders, None), lambda ign: None)
        d = protocol._responseDeferred
        protocol.makeConnection(StringTransport())
        protocol.connectionLost(ConnectionLost())
        return self.assertFailure(d, ResponseNeverReceived)

    def test_someResponseButNotAll(self):
        """
        If a partial response was received and the connection is lost, the
        resulting error is L{ResponseFailed}, but not
        L{ResponseNeverReceived}.
        """
        protocol = HTTPClientParser(Request(b'HEAD', b'/', _boringHeaders, None), lambda ign: None)
        d = protocol._responseDeferred
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'2')
        protocol.connectionLost(ConnectionLost())
        return self.assertFailure(d, ResponseFailed).addCallback(self.assertIsInstance, ResponseFailed)

    def test_1XXResponseIsSwallowed(self):
        """
        If a response in the 1XX range is received it just gets swallowed and
        the parser resets itself.
        """
        sample103Response = b'HTTP/1.1 103 Early Hints\r\nServer: socketserver/1.0.0\r\nLink: </other/styles.css>; rel=preload; as=style\r\nLink: </other/action.js>; rel=preload; as=script\r\n\r\n'
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda ign: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(sample103Response)
        self.assertTrue(getattr(protocol, 'response', None) is None)
        self.assertEqual(protocol.state, STATUS)
        self.assertEqual(len(list(protocol.headers.getAllRawHeaders())), 0)
        self.assertEqual(len(list(protocol.connHeaders.getAllRawHeaders())), 0)
        self.assertTrue(protocol._everReceivedData)

    def test_1XXFollowedByFinalResponseOnlyEmitsFinal(self):
        """
        When a 1XX response is swallowed, the final response that follows it is
        the only one that gets sent to the application.
        """
        sample103Response = b'HTTP/1.1 103 Early Hints\r\nServer: socketserver/1.0.0\r\nLink: </other/styles.css>; rel=preload; as=style\r\nLink: </other/action.js>; rel=preload; as=script\r\n\r\n'
        following200Response = b'HTTP/1.1 200 OK\r\nContent-Length: 123\r\n\r\n'
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda ign: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(sample103Response + following200Response)
        self.assertEqual(protocol.response.code, 200)
        self.assertEqual(protocol.response.headers, Headers({}))
        self.assertEqual(protocol.connHeaders, Headers({b'content-length': [b'123']}))
        self.assertEqual(protocol.response.length, 123)

    def test_multiple1XXResponsesAreIgnored(self):
        """
        It is acceptable for multiple 1XX responses to come through, all of
        which get ignored.
        """
        sample103Response = b'HTTP/1.1 103 Early Hints\r\nServer: socketserver/1.0.0\r\nLink: </other/styles.css>; rel=preload; as=style\r\nLink: </other/action.js>; rel=preload; as=script\r\n\r\n'
        following200Response = b'HTTP/1.1 200 OK\r\nContent-Length: 123\r\n\r\n'
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda ign: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(sample103Response + sample103Response + sample103Response + following200Response)
        self.assertEqual(protocol.response.code, 200)
        self.assertEqual(protocol.response.headers, Headers({}))
        self.assertEqual(protocol.connHeaders, Headers({b'content-length': [b'123']}))
        self.assertEqual(protocol.response.length, 123)

    def test_ignored1XXResponseCausesLog(self):
        """
        When a 1XX response is ignored, Twisted emits a log.
        """
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        sample103Response = b'HTTP/1.1 103 Early Hints\r\nServer: socketserver/1.0.0\r\nLink: </other/styles.css>; rel=preload; as=style\r\nLink: </other/action.js>; rel=preload; as=script\r\n\r\n'
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda ign: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(sample103Response)
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        self.assertEquals(event['log_format'], 'Ignoring unexpected {code} response')
        self.assertEquals(event['code'], 103)