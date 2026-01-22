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
class IdentityTransferEncodingTests(TestCase):
    """
    Tests for L{_IdentityTransferDecoder}.
    """

    def setUp(self):
        """
        Create an L{_IdentityTransferDecoder} with callbacks hooked up so that
        calls to them can be inspected.
        """
        self.data = []
        self.finish = []
        self.contentLength = 10
        self.decoder = _IdentityTransferDecoder(self.contentLength, self.data.append, self.finish.append)

    def test_exactAmountReceived(self):
        """
        If L{_IdentityTransferDecoder.dataReceived} is called with a byte string
        with length equal to the content length passed to
        L{_IdentityTransferDecoder}'s initializer, the data callback is invoked
        with that string and the finish callback is invoked with a zero-length
        string.
        """
        self.decoder.dataReceived(b'x' * self.contentLength)
        self.assertEqual(self.data, [b'x' * self.contentLength])
        self.assertEqual(self.finish, [b''])

    def test_shortStrings(self):
        """
        If L{_IdentityTransferDecoder.dataReceived} is called multiple times
        with byte strings which, when concatenated, are as long as the content
        length provided, the data callback is invoked with each string and the
        finish callback is invoked only after the second call.
        """
        self.decoder.dataReceived(b'x')
        self.assertEqual(self.data, [b'x'])
        self.assertEqual(self.finish, [])
        self.decoder.dataReceived(b'y' * (self.contentLength - 1))
        self.assertEqual(self.data, [b'x', b'y' * (self.contentLength - 1)])
        self.assertEqual(self.finish, [b''])

    def test_longString(self):
        """
        If L{_IdentityTransferDecoder.dataReceived} is called with a byte string
        with length greater than the provided content length, only the prefix
        of that string up to the content length is passed to the data callback
        and the remainder is passed to the finish callback.
        """
        self.decoder.dataReceived(b'x' * self.contentLength + b'y')
        self.assertEqual(self.data, [b'x' * self.contentLength])
        self.assertEqual(self.finish, [b'y'])

    def test_rejectDataAfterFinished(self):
        """
        If data is passed to L{_IdentityTransferDecoder.dataReceived} after the
        finish callback has been invoked, C{RuntimeError} is raised.
        """
        failures = []

        def finish(bytes):
            try:
                decoder.dataReceived(b'foo')
            except BaseException:
                failures.append(Failure())
        decoder = _IdentityTransferDecoder(5, self.data.append, finish)
        decoder.dataReceived(b'x' * 4)
        self.assertEqual(failures, [])
        decoder.dataReceived(b'y')
        failures[0].trap(RuntimeError)
        self.assertEqual(str(failures[0].value), '_IdentityTransferDecoder cannot decode data after finishing')

    def test_unknownContentLength(self):
        """
        If L{_IdentityTransferDecoder} is constructed with L{None} for the
        content length, it passes all data delivered to it through to the data
        callback.
        """
        data = []
        finish = []
        decoder = _IdentityTransferDecoder(None, data.append, finish.append)
        decoder.dataReceived(b'x')
        self.assertEqual(data, [b'x'])
        decoder.dataReceived(b'y')
        self.assertEqual(data, [b'x', b'y'])
        self.assertEqual(finish, [])

    def _verifyCallbacksUnreferenced(self, decoder):
        """
        Check the decoder's data and finish callbacks and make sure they are
        None in order to help avoid references cycles.
        """
        self.assertIdentical(decoder.dataCallback, None)
        self.assertIdentical(decoder.finishCallback, None)

    def test_earlyConnectionLose(self):
        """
        L{_IdentityTransferDecoder.noMoreData} raises L{_DataLoss} if it is
        called and the content length is known but not enough bytes have been
        delivered.
        """
        self.decoder.dataReceived(b'x' * (self.contentLength - 1))
        self.assertRaises(_DataLoss, self.decoder.noMoreData)
        self._verifyCallbacksUnreferenced(self.decoder)

    def test_unknownContentLengthConnectionLose(self):
        """
        L{_IdentityTransferDecoder.noMoreData} calls the finish callback and
        raises L{PotentialDataLoss} if it is called and the content length is
        unknown.
        """
        body = []
        finished = []
        decoder = _IdentityTransferDecoder(None, body.append, finished.append)
        self.assertRaises(PotentialDataLoss, decoder.noMoreData)
        self.assertEqual(body, [])
        self.assertEqual(finished, [b''])
        self._verifyCallbacksUnreferenced(decoder)

    def test_finishedConnectionLose(self):
        """
        L{_IdentityTransferDecoder.noMoreData} does not raise any exception if
        it is called when the content length is known and that many bytes have
        been delivered.
        """
        self.decoder.dataReceived(b'x' * self.contentLength)
        self.decoder.noMoreData()
        self._verifyCallbacksUnreferenced(self.decoder)