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
class SingleRangeStaticProducerTests(TestCase):
    """
    Tests for L{SingleRangeStaticProducer}.
    """

    def test_implementsIPullProducer(self):
        """
        L{SingleRangeStaticProducer} implements L{IPullProducer}.
        """
        verifyObject(interfaces.IPullProducer, static.SingleRangeStaticProducer(None, None, None, None))

    def test_resumeProducingProducesContent(self):
        """
        L{SingleRangeStaticProducer.resumeProducing} writes the given amount
        of content, starting at the given offset, from the resource to the
        request.
        """
        request = DummyRequest([])
        content = b'abcdef'
        producer = static.SingleRangeStaticProducer(request, StringIO(content), 1, 3)
        producer.start()
        self.assertEqual(content[1:4], b''.join(request.written))

    def test_resumeProducingBuffersOutput(self):
        """
        L{SingleRangeStaticProducer.start} writes at most
        C{abstract.FileDescriptor.bufferSize} bytes of content from the
        resource to the request at once.
        """
        request = DummyRequest([])
        bufferSize = abstract.FileDescriptor.bufferSize
        content = b'abc' * bufferSize
        producer = static.SingleRangeStaticProducer(request, StringIO(content), 1, bufferSize + 10)
        producer.start()
        expected = [content[1:bufferSize + 1], content[bufferSize + 1:bufferSize + 11]]
        self.assertEqual(expected, request.written)

    def test_finishCalledWhenDone(self):
        """
        L{SingleRangeStaticProducer.resumeProducing} calls finish() on the
        request after it is done producing content.
        """
        request = DummyRequest([])
        finishDeferred = request.notifyFinish()
        callbackList = []
        finishDeferred.addCallback(callbackList.append)
        producer = static.SingleRangeStaticProducer(request, StringIO(b'abcdef'), 1, 1)
        producer.start()
        self.assertEqual([None], callbackList)