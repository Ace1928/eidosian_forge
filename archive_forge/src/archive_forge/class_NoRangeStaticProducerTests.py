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
class NoRangeStaticProducerTests(TestCase):
    """
    Tests for L{NoRangeStaticProducer}.
    """

    def test_implementsIPullProducer(self):
        """
        L{NoRangeStaticProducer} implements L{IPullProducer}.
        """
        verifyObject(interfaces.IPullProducer, static.NoRangeStaticProducer(None, None))

    def test_resumeProducingProducesContent(self):
        """
        L{NoRangeStaticProducer.resumeProducing} writes content from the
        resource to the request.
        """
        request = DummyRequest([])
        content = b'abcdef'
        producer = static.NoRangeStaticProducer(request, StringIO(content))
        producer.start()
        self.assertEqual(content, b''.join(request.written))

    def test_resumeProducingBuffersOutput(self):
        """
        L{NoRangeStaticProducer.start} writes at most
        C{abstract.FileDescriptor.bufferSize} bytes of content from the
        resource to the request at once.
        """
        request = DummyRequest([])
        bufferSize = abstract.FileDescriptor.bufferSize
        content = b'a' * (2 * bufferSize + 1)
        producer = static.NoRangeStaticProducer(request, StringIO(content))
        producer.start()
        expected = [content[0:bufferSize], content[bufferSize:2 * bufferSize], content[2 * bufferSize:]]
        self.assertEqual(expected, request.written)

    def test_finishCalledWhenDone(self):
        """
        L{NoRangeStaticProducer.resumeProducing} calls finish() on the request
        after it is done producing content.
        """
        request = DummyRequest([])
        finishDeferred = request.notifyFinish()
        callbackList = []
        finishDeferred.addCallback(callbackList.append)
        producer = static.NoRangeStaticProducer(request, StringIO(b'abcdef'))
        producer.start()
        self.assertEqual([None], callbackList)