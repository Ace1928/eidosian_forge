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
class ChunkedTransferEncodingTests(unittest.TestCase):
    """
    Tests for L{_ChunkedTransferDecoder}, which turns a byte stream encoded
    using HTTP I{chunked} C{Transfer-Encoding} back into the original byte
    stream.
    """

    def test_decoding(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} decodes chunked-encoded data
        and passes the result to the specified callback.
        """
        L = []
        p = http._ChunkedTransferDecoder(L.append, None)
        p.dataReceived(b'3\r\nabc\r\n5\r\n12345\r\n')
        p.dataReceived(b'a\r\n0123456789\r\n')
        self.assertEqual(L, [b'abc', b'12345', b'0123456789'])

    def test_short(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} decodes chunks broken up and
        delivered in multiple calls.
        """
        L = []
        finished = []
        p = http._ChunkedTransferDecoder(L.append, finished.append)
        for s in iterbytes(b'3\r\nabc\r\n5\r\n12345\r\n0\r\n\r\n'):
            p.dataReceived(s)
        self.assertEqual(L, [b'a', b'b', b'c', b'1', b'2', b'3', b'4', b'5'])
        self.assertEqual(finished, [b''])
        self.assertEqual(p._trailerHeaders, [])

    def test_long(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} delivers partial chunk data as
        soon as it is received.
        """
        data = []
        finished = []
        p = http._ChunkedTransferDecoder(data.append, finished.append)
        p.dataReceived(b'a;\r\n12345')
        p.dataReceived(b'67890')
        p.dataReceived(b'\r\n0;\r\n\r\n...')
        self.assertEqual(data, [b'12345', b'67890'])
        self.assertEqual(finished, [b'...'])

    def test_empty(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} is robust against receiving
        a zero-length input.
        """
        chunks = []
        finished = []
        p = http._ChunkedTransferDecoder(chunks.append, finished.append)
        p.dataReceived(b'')
        for s in iterbytes(b'3\r\nabc\r\n5\r\n12345\r\n0\r\n\r\n'):
            p.dataReceived(s)
            p.dataReceived(b'')
        self.assertEqual(chunks, [b'a', b'b', b'c', b'1', b'2', b'3', b'4', b'5'])
        self.assertEqual(finished, [b''])

    def test_newlines(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} doesn't treat CR LF pairs
        embedded in chunk bodies specially.
        """
        L = []
        p = http._ChunkedTransferDecoder(L.append, None)
        p.dataReceived(b'2\r\n\r\n\r\n')
        self.assertEqual(L, [b'\r\n'])

    def test_extensions(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} disregards chunk-extension
        fields.
        """
        L = []
        p = http._ChunkedTransferDecoder(L.append, None)
        p.dataReceived(b'3; x-foo=bar\r\nabc\r\n')
        self.assertEqual(L, [b'abc'])

    def test_extensionsMalformed(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} raises
        L{_MalformedChunkedDataError} when the chunk extension fields contain
        invalid characters.

        This is a potential request smuggling vector: see GHSA-c2jg-hw38-jrqq.
        """
        invalidControl = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        invalidDelimiter = b'\\'
        invalidDel = b'\x7f'
        for b in invalidControl + invalidDelimiter + invalidDel:
            data = b'3; ' + bytes((b,)) + b'\r\nabc\r\n'
            p = http._ChunkedTransferDecoder(lambda b: None, lambda b: None)
            self.assertRaises(http._MalformedChunkedDataError, p.dataReceived, data)

    def test_oversizedChunkSizeLine(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} raises
        L{_MalformedChunkedDataError} when the chunk size line exceeds 4 KiB.
        This applies even when the data has already been received and buffered
        so that behavior is consistent regardless of how bytes are framed.
        """
        p = http._ChunkedTransferDecoder(None, None)
        self.assertRaises(http._MalformedChunkedDataError, p.dataReceived, b'3;' + b'.' * http.maxChunkSizeLineLength + b'\r\nabc\r\n')

    def test_oversizedChunkSizeLinePartial(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} raises
        L{_MalformedChunkedDataError} when the amount of data buffered while
        looking for the end of the chunk size line exceeds 4 KiB so
        that buffering does not continue without bound.
        """
        p = http._ChunkedTransferDecoder(None, None)
        self.assertRaises(http._MalformedChunkedDataError, p.dataReceived, b'.' * (http.maxChunkSizeLineLength + 1))

    def test_malformedChunkSize(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} raises
        L{_MalformedChunkedDataError} when the chunk size can't be decoded as
        a base-16 integer.
        """
        p = http._ChunkedTransferDecoder(lambda b: None, lambda b: None)
        self.assertRaises(http._MalformedChunkedDataError, p.dataReceived, b'bloop\r\nabc\r\n')

    def test_malformedChunkSizeNegative(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} raises
        L{_MalformedChunkedDataError} when the chunk size is negative.
        """
        p = http._ChunkedTransferDecoder(lambda b: None, lambda b: None)
        self.assertRaises(http._MalformedChunkedDataError, p.dataReceived, b'-3\r\nabc\r\n')

    def test_malformedChunkSizeHex(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} raises
        L{_MalformedChunkedDataError} when the chunk size is prefixed with
        "0x", as if it were a Python integer literal.

        This is a potential request smuggling vector: see GHSA-c2jg-hw38-jrqq.
        """
        p = http._ChunkedTransferDecoder(lambda b: None, lambda b: None)
        self.assertRaises(http._MalformedChunkedDataError, p.dataReceived, b'0x3\r\nabc\r\n')

    def test_malformedChunkEnd(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} raises
        L{_MalformedChunkedDataError} when the chunk is followed by characters
        other than C{\\r\\n}.
        """
        p = http._ChunkedTransferDecoder(lambda b: None, lambda b: None)
        self.assertRaises(http._MalformedChunkedDataError, p.dataReceived, b'3\r\nabc!!!!')

    def test_finish(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} interprets a zero-length
        chunk as the end of the chunked data stream and calls the completion
        callback.
        """
        finished = []
        p = http._ChunkedTransferDecoder(None, finished.append)
        p.dataReceived(b'0\r\n\r\n')
        self.assertEqual(finished, [b''])

    def test_extra(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} passes any bytes which come
        after the terminating zero-length chunk to the completion callback.
        """
        finished = []
        p = http._ChunkedTransferDecoder(None, finished.append)
        p.dataReceived(b'0\r\n\r\nhello')
        self.assertEqual(finished, [b'hello'])

    def test_afterFinished(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} raises C{RuntimeError} if it
        is called after it has seen the last chunk.
        """
        p = http._ChunkedTransferDecoder(None, lambda bytes: None)
        p.dataReceived(b'0\r\n\r\n')
        self.assertRaises(RuntimeError, p.dataReceived, b'hello')

    def test_earlyConnectionLose(self):
        """
        L{_ChunkedTransferDecoder.noMoreData} raises L{_DataLoss} if it is
        called and the end of the last trailer has not yet been received.
        """
        parser = http._ChunkedTransferDecoder(None, lambda bytes: None)
        parser.dataReceived(b'0\r\n\r')
        exc = self.assertRaises(_DataLoss, parser.noMoreData)
        self.assertEqual(str(exc), "Chunked decoder in 'TRAILER' state, still expecting more data to get to 'FINISHED' state.")

    def test_finishedConnectionLose(self):
        """
        L{_ChunkedTransferDecoder.noMoreData} does not raise any exception if
        it is called after the terminal zero length chunk is received.
        """
        parser = http._ChunkedTransferDecoder(None, lambda bytes: None)
        parser.dataReceived(b'0\r\n\r\n')
        parser.noMoreData()

    def test_reentrantFinishedNoMoreData(self):
        """
        L{_ChunkedTransferDecoder.noMoreData} can be called from the finished
        callback without raising an exception.
        """
        errors = []
        successes = []

        def finished(extra):
            try:
                parser.noMoreData()
            except BaseException:
                errors.append(Failure())
            else:
                successes.append(True)
        parser = http._ChunkedTransferDecoder(None, finished)
        parser.dataReceived(b'0\r\n\r\n')
        self.assertEqual(errors, [])
        self.assertEqual(successes, [True])

    def test_trailerHeaders(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} decodes chunked-encoded data
        and ignores trailer headers which come after the terminating zero-length
        chunk.
        """
        L = []
        finished = []
        p = http._ChunkedTransferDecoder(L.append, finished.append)
        p.dataReceived(b'3\r\nabc\r\n5\r\n12345\r\n')
        p.dataReceived(b'a\r\n0123456789\r\n0\r\nServer-Timing: total;dur=123.4\r\nExpires: Wed, 21 Oct 2015 07:28:00 GMT\r\n\r\n')
        self.assertEqual(L, [b'abc', b'12345', b'0123456789'])
        self.assertEqual(finished, [b''])
        self.assertEqual(p._trailerHeaders, [b'Server-Timing: total;dur=123.4', b'Expires: Wed, 21 Oct 2015 07:28:00 GMT'])

    def test_shortTrailerHeader(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} decodes chunks of input with
        tailer header broken up and delivered in multiple calls.
        """
        L = []
        finished = []
        p = http._ChunkedTransferDecoder(L.append, finished.append)
        for s in iterbytes(b'3\r\nabc\r\n5\r\n12345\r\n0\r\nServer-Timing: total;dur=123.4\r\n\r\n'):
            p.dataReceived(s)
        self.assertEqual(L, [b'a', b'b', b'c', b'1', b'2', b'3', b'4', b'5'])
        self.assertEqual(finished, [b''])
        self.assertEqual(p._trailerHeaders, [b'Server-Timing: total;dur=123.4'])

    def test_tooLongTrailerHeader(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} raises
        L{_MalformedChunkedDataError} when the trailing headers data is too long.
        """
        p = http._ChunkedTransferDecoder(lambda b: None, lambda b: None)
        p._maxTrailerHeadersSize = 10
        self.assertRaises(http._MalformedChunkedDataError, p.dataReceived, b'3\r\nabc\r\n0\r\nTotal-Trailer: header;greater-then=10\r\n\r\n')

    def test_unfinishedTrailerHeader(self):
        """
        L{_ChunkedTransferDecoder.dataReceived} raises
        L{_MalformedChunkedDataError} when the trailing headers data is too long
        and doesn't have final CRLF characters.
        """
        p = http._ChunkedTransferDecoder(lambda b: None, lambda b: None)
        p._maxTrailerHeadersSize = 10
        p.dataReceived(b'3\r\nabc\r\n0\r\n0123456789')
        self.assertRaises(http._MalformedChunkedDataError, p.dataReceived, b'A')