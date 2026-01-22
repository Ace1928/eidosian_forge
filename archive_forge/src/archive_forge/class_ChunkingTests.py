import base64
import calendar
import random
from io import BytesIO
from itertools import cycle
from typing import Sequence, Union
from unittest import skipIf
from urllib.parse import clear_cache  # type: ignore[attr-defined]
from urllib.parse import urlparse, urlunsplit
from zope.interface import directlyProvides, providedBy, provider
from zope.interface.verify import verifyObject
import hamcrest
from twisted.internet import address
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols import loopback
from twisted.python.compat import iterbytes, networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
from twisted.web import http, http_headers, iweb
from twisted.web.http import PotentialDataLoss, _DataLoss, _IdentityTransferDecoder
from twisted.web.test.requesthelper import (
from ._util import assertIsFilesystemTemporary
class ChunkingTests(unittest.TestCase, ResponseTestMixin):
    strings = [b'abcv', b'', b'fdfsd423', b'Ffasfas\r\n', b'523523\n\rfsdf', b'4234']

    def testChunks(self):
        for s in self.strings:
            chunked = b''.join(http.toChunk(s))
            self.assertEqual((s, b''), http.fromChunk(chunked))
        self.assertRaises(ValueError, http.fromChunk, b'-5\r\nmalformed!\r\n')
        self.assertRaises(ValueError, http.fromChunk, b'0xa\r\nmalformed!\r\n')
        self.assertRaises(ValueError, http.fromChunk, b'0XA\r\nmalformed!\r\n')

    def testConcatenatedChunks(self):
        chunked = b''.join([b''.join(http.toChunk(t)) for t in self.strings])
        result = []
        buffer = b''
        for c in iterbytes(chunked):
            buffer = buffer + c
            try:
                data, buffer = http.fromChunk(buffer)
                result.append(data)
            except ValueError:
                pass
        self.assertEqual(result, self.strings)

    def test_chunkedResponses(self):
        """
        Test that the L{HTTPChannel} correctly chunks responses when needed.
        """
        trans = StringTransport()
        channel = http.HTTPChannel()
        channel.makeConnection(trans)
        req = http.Request(channel, False)
        req.setResponseCode(200)
        req.clientproto = b'HTTP/1.1'
        req.responseHeaders.setRawHeaders(b'test', [b'lemur'])
        req.write(b'Hello')
        req.write(b'World!')
        self.assertResponseEquals(trans.value(), [(b'HTTP/1.1 200 OK', b'Test: lemur', b'Transfer-Encoding: chunked', b'5\r\nHello\r\n6\r\nWorld!\r\n')])

    def runChunkedRequest(self, httpRequest, requestFactory=None, chunkSize=1):
        """
        Execute a web request based on plain text content, chunking
        the request payload.

        This is a stripped-down, chunking version of ParsingTests.runRequest.
        """
        channel = http.HTTPChannel()
        if requestFactory:
            channel.requestFactory = _makeRequestProxyFactory(requestFactory)
        httpRequest = httpRequest.replace(b'\n', b'\r\n')
        header, body = httpRequest.split(b'\r\n\r\n', 1)
        transport = StringTransport()
        channel.makeConnection(transport)
        channel.dataReceived(header + b'\r\n\r\n')
        for pos in range(len(body) // chunkSize + 1):
            if channel.transport.disconnecting:
                break
            channel.dataReceived(b''.join(http.toChunk(body[pos * chunkSize:(pos + 1) * chunkSize])))
        channel.dataReceived(b''.join(http.toChunk(b'')))
        channel.connectionLost(IOError('all done'))
        return channel

    def test_multipartFormData(self):
        """
        Test that chunked uploads are actually processed into args.

        This is essentially a copy of ParsingTests.test_multipartFormData,
        just with chunking put in.

        This fails as of twisted version 18.9.0 because of bug #9678.
        """
        processed = []

        class MyRequest(http.Request):

            def process(self):
                processed.append(self)
                self.write(b'done')
                self.finish()
        req = b'POST / HTTP/1.0\nContent-Type: multipart/form-data; boundary=AaB03x\nTransfer-Encoding: chunked\n\n--AaB03x\nContent-Type: text/plain\nContent-Disposition: form-data; name="text"\nContent-Transfer-Encoding: quoted-printable\n\nabasdfg\n--AaB03x--\n'
        channel = self.runChunkedRequest(req, MyRequest, chunkSize=5)
        self.assertEqual(channel.transport.value(), b'HTTP/1.0 200 OK\r\n\r\ndone')
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].args, {b'text': [b'abasdfg']})