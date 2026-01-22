import errno
import inspect
import mimetypes
import os
import re
import sys
import warnings
from io import BytesIO as StringIO
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import abstract, interfaces
from twisted.python import compat, log
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.web import http, resource, script, static
from twisted.web._responses import FOUND
from twisted.web.server import UnsupportedMethod
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyRequest
class RangeTests(TestCase):
    """
    Tests for I{Range-Header} support in L{twisted.web.static.File}.

    @type file: L{file}
    @ivar file: Temporary (binary) file containing the content to be served.

    @type resource: L{static.File}
    @ivar resource: A leaf web resource using C{file} as content.

    @type request: L{DummyRequest}
    @ivar request: A fake request, requesting C{resource}.

    @type catcher: L{list}
    @ivar catcher: List which gathers all log information.
    """

    def setUp(self):
        """
        Create a temporary file with a fixed payload of 64 bytes.  Create a
        resource for that file and create a request which will be for that
        resource.  Each test can set a different range header to test different
        aspects of the implementation.
        """
        path = FilePath(self.mktemp())
        self.payload = b'\xf8u\xf3E\x8c7\xce\x00\x9e\xb6a0y0S\xf0\xef\xac\xb7\xbe\xb5\x17M\x1e\x136k{\x1e\xbe\x0c\x07\x07\t\xd0\xbckY\xf5I\x0b\xb8\x88oZ\x1d\x85b\x1a\xcdk\xf2\x1d&\xfd%\xdd\x82q/A\x10Y\x8b'
        path.setContent(self.payload)
        self.file = path.open()
        self.resource = static.File(self.file.name)
        self.resource.isLeaf = 1
        self.request = DummyRequest([b''])
        self.request.uri = self.file.name
        self.catcher = []
        log.addObserver(self.catcher.append)

    def tearDown(self):
        """
        Clean up the resource file and the log observer.
        """
        self.file.close()
        log.removeObserver(self.catcher.append)

    def _assertLogged(self, expected):
        """
        Asserts that a given log message occurred with an expected message.
        """
        logItem = self.catcher.pop()
        self.assertEqual(logItem['message'][0], expected)
        self.assertEqual(self.catcher, [], f'An additional log occurred: {logItem!r}')

    def test_invalidRanges(self):
        """
        L{File._parseRangeHeader} raises L{ValueError} when passed
        syntactically invalid byte ranges.
        """
        f = self.resource._parseRangeHeader
        self.assertRaises(ValueError, f, b'bytes')
        self.assertRaises(ValueError, f, b'unknown=1-2')
        self.assertRaises(ValueError, f, b'bytes=3')
        self.assertRaises(ValueError, f, b'bytes=-')
        self.assertRaises(ValueError, f, b'bytes=foo-')
        self.assertRaises(ValueError, f, b'bytes=-foo')
        self.assertRaises(ValueError, f, b'bytes=5-4')

    def test_rangeMissingStop(self):
        """
        A single bytes range without an explicit stop position is parsed into a
        two-tuple giving the start position and L{None}.
        """
        self.assertEqual(self.resource._parseRangeHeader(b'bytes=0-'), [(0, None)])

    def test_rangeMissingStart(self):
        """
        A single bytes range without an explicit start position is parsed into
        a two-tuple of L{None} and the end position.
        """
        self.assertEqual(self.resource._parseRangeHeader(b'bytes=-3'), [(None, 3)])

    def test_range(self):
        """
        A single bytes range with explicit start and stop positions is parsed
        into a two-tuple of those positions.
        """
        self.assertEqual(self.resource._parseRangeHeader(b'bytes=2-5'), [(2, 5)])

    def test_rangeWithSpace(self):
        """
        A single bytes range with whitespace in allowed places is parsed in
        the same way as it would be without the whitespace.
        """
        self.assertEqual(self.resource._parseRangeHeader(b' bytes=1-2 '), [(1, 2)])
        self.assertEqual(self.resource._parseRangeHeader(b'bytes =1-2 '), [(1, 2)])
        self.assertEqual(self.resource._parseRangeHeader(b'bytes= 1-2'), [(1, 2)])
        self.assertEqual(self.resource._parseRangeHeader(b'bytes=1 -2'), [(1, 2)])
        self.assertEqual(self.resource._parseRangeHeader(b'bytes=1- 2'), [(1, 2)])
        self.assertEqual(self.resource._parseRangeHeader(b'bytes=1-2 '), [(1, 2)])

    def test_nullRangeElements(self):
        """
        If there are multiple byte ranges but only one is non-null, the
        non-null range is parsed and its start and stop returned.
        """
        self.assertEqual(self.resource._parseRangeHeader(b'bytes=1-2,\r\n, ,\t'), [(1, 2)])

    def test_multipleRanges(self):
        """
        If multiple byte ranges are specified their starts and stops are
        returned.
        """
        self.assertEqual(self.resource._parseRangeHeader(b'bytes=1-2,3-4'), [(1, 2), (3, 4)])

    def test_bodyLength(self):
        """
        A correct response to a range request is as long as the length of the
        requested range.
        """
        self.request.requestHeaders.addRawHeader(b'range', b'bytes=0-43')
        self.resource.render(self.request)
        self.assertEqual(len(b''.join(self.request.written)), 44)

    def test_invalidRangeRequest(self):
        """
        An incorrect range request (RFC 2616 defines a correct range request as
        a Bytes-Unit followed by a '=' character followed by a specific range.
        Only 'bytes' is defined) results in the range header value being logged
        and a normal 200 response being sent.
        """
        range = b'foobar=0-43'
        self.request.requestHeaders.addRawHeader(b'range', range)
        self.resource.render(self.request)
        expected = f'Ignoring malformed Range header {range.decode()!r}'
        self._assertLogged(expected)
        self.assertEqual(b''.join(self.request.written), self.payload)
        self.assertEqual(self.request.responseCode, http.OK)
        self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-length')[0], b'%d' % (len(self.payload),))

    def parseMultipartBody(self, body, boundary):
        """
        Parse C{body} as a multipart MIME response separated by C{boundary}.

        Note that this with fail the calling test on certain syntactic
        problems.
        """
        sep = b'\r\n--' + boundary
        parts = body.split(sep)
        self.assertEqual(b'', parts[0])
        self.assertEqual(b'--\r\n', parts[-1])
        parsed_parts = []
        for part in parts[1:-1]:
            before, header1, header2, blank, partBody = part.split(b'\r\n', 4)
            headers = header1 + b'\n' + header2
            self.assertEqual(b'', before)
            self.assertEqual(b'', blank)
            partContentTypeValue = re.search(b'^content-type: (.*)$', headers, re.I | re.M).group(1)
            start, end, size = re.search(b'^content-range: bytes ([0-9]+)-([0-9]+)/([0-9]+)$', headers, re.I | re.M).groups()
            parsed_parts.append({b'contentType': partContentTypeValue, b'contentRange': (start, end, size), b'body': partBody})
        return parsed_parts

    def test_multipleRangeRequest(self):
        """
        The response to a request for multiple bytes ranges is a MIME-ish
        multipart response.
        """
        startEnds = [(0, 2), (20, 30), (40, 50)]
        rangeHeaderValue = b','.join([networkString(f'{s}-{e}') for s, e in startEnds])
        self.request.requestHeaders.addRawHeader(b'range', b'bytes=' + rangeHeaderValue)
        self.resource.render(self.request)
        self.assertEqual(self.request.responseCode, http.PARTIAL_CONTENT)
        boundary = re.match(b'^multipart/byteranges; boundary="(.*)"$', self.request.responseHeaders.getRawHeaders(b'content-type')[0]).group(1)
        parts = self.parseMultipartBody(b''.join(self.request.written), boundary)
        self.assertEqual(len(startEnds), len(parts))
        for part, (s, e) in zip(parts, startEnds):
            self.assertEqual(networkString(self.resource.type), part[b'contentType'])
            start, end, size = part[b'contentRange']
            self.assertEqual(int(start), s)
            self.assertEqual(int(end), e)
            self.assertEqual(int(size), self.resource.getFileSize())
            self.assertEqual(self.payload[s:e + 1], part[b'body'])

    def test_multipleRangeRequestWithRangeOverlappingEnd(self):
        """
        The response to a request for multiple bytes ranges is a MIME-ish
        multipart response, even when one of the ranged falls off the end of
        the resource.
        """
        startEnds = [(0, 2), (40, len(self.payload) + 10)]
        rangeHeaderValue = b','.join([networkString(f'{s}-{e}') for s, e in startEnds])
        self.request.requestHeaders.addRawHeader(b'range', b'bytes=' + rangeHeaderValue)
        self.resource.render(self.request)
        self.assertEqual(self.request.responseCode, http.PARTIAL_CONTENT)
        boundary = re.match(b'^multipart/byteranges; boundary="(.*)"$', self.request.responseHeaders.getRawHeaders(b'content-type')[0]).group(1)
        parts = self.parseMultipartBody(b''.join(self.request.written), boundary)
        self.assertEqual(len(startEnds), len(parts))
        for part, (s, e) in zip(parts, startEnds):
            self.assertEqual(networkString(self.resource.type), part[b'contentType'])
            start, end, size = part[b'contentRange']
            self.assertEqual(int(start), s)
            self.assertEqual(int(end), min(e, self.resource.getFileSize() - 1))
            self.assertEqual(int(size), self.resource.getFileSize())
            self.assertEqual(self.payload[s:e + 1], part[b'body'])

    def test_implicitEnd(self):
        """
        If the end byte position is omitted, then it is treated as if the
        length of the resource was specified by the end byte position.
        """
        self.request.requestHeaders.addRawHeader(b'range', b'bytes=23-')
        self.resource.render(self.request)
        self.assertEqual(b''.join(self.request.written), self.payload[23:])
        self.assertEqual(len(b''.join(self.request.written)), 41)
        self.assertEqual(self.request.responseCode, http.PARTIAL_CONTENT)
        self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-range')[0], b'bytes 23-63/64')
        self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-length')[0], b'41')

    def test_implicitStart(self):
        """
        If the start byte position is omitted but the end byte position is
        supplied, then the range is treated as requesting the last -N bytes of
        the resource, where N is the end byte position.
        """
        self.request.requestHeaders.addRawHeader(b'range', b'bytes=-17')
        self.resource.render(self.request)
        self.assertEqual(b''.join(self.request.written), self.payload[-17:])
        self.assertEqual(len(b''.join(self.request.written)), 17)
        self.assertEqual(self.request.responseCode, http.PARTIAL_CONTENT)
        self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-range')[0], b'bytes 47-63/64')
        self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-length')[0], b'17')

    def test_explicitRange(self):
        """
        A correct response to a bytes range header request from A to B starts
        with the A'th byte and ends with (including) the B'th byte. The first
        byte of a page is numbered with 0.
        """
        self.request.requestHeaders.addRawHeader(b'range', b'bytes=3-43')
        self.resource.render(self.request)
        written = b''.join(self.request.written)
        self.assertEqual(written, self.payload[3:44])
        self.assertEqual(self.request.responseCode, http.PARTIAL_CONTENT)
        self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-range')[0], b'bytes 3-43/64')
        self.assertEqual(b'%d' % (len(written),), self.request.responseHeaders.getRawHeaders(b'content-length')[0])

    def test_explicitRangeOverlappingEnd(self):
        """
        A correct response to a bytes range header request from A to B when B
        is past the end of the resource starts with the A'th byte and ends
        with the last byte of the resource. The first byte of a page is
        numbered with 0.
        """
        self.request.requestHeaders.addRawHeader(b'range', b'bytes=40-100')
        self.resource.render(self.request)
        written = b''.join(self.request.written)
        self.assertEqual(written, self.payload[40:])
        self.assertEqual(self.request.responseCode, http.PARTIAL_CONTENT)
        self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-range')[0], b'bytes 40-63/64')
        self.assertEqual(b'%d' % (len(written),), self.request.responseHeaders.getRawHeaders(b'content-length')[0])

    def test_statusCodeRequestedRangeNotSatisfiable(self):
        """
        If a range is syntactically invalid due to the start being greater than
        the end, the range header is ignored (the request is responded to as if
        it were not present).
        """
        self.request.requestHeaders.addRawHeader(b'range', b'bytes=20-13')
        self.resource.render(self.request)
        self.assertEqual(self.request.responseCode, http.OK)
        self.assertEqual(b''.join(self.request.written), self.payload)
        self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-length')[0], b'%d' % (len(self.payload),))

    def test_invalidStartBytePos(self):
        """
        If a range is unsatisfiable due to the start not being less than the
        length of the resource, the response is 416 (Requested range not
        satisfiable) and no data is written to the response body (RFC 2616,
        section 14.35.1).
        """
        self.request.requestHeaders.addRawHeader(b'range', b'bytes=67-108')
        self.resource.render(self.request)
        self.assertEqual(self.request.responseCode, http.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.assertEqual(b''.join(self.request.written), b'')
        self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-length')[0], b'0')
        self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-range')[0], networkString('bytes */%d' % (len(self.payload),)))