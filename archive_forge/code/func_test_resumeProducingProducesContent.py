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
def test_resumeProducingProducesContent(self):
    """
        L{MultipleRangeStaticProducer.resumeProducing} writes the requested
        chunks of content from the resource to the request, with the supplied
        boundaries in between each chunk.
        """
    request = DummyRequest([])
    content = b'abcdef'
    producer = static.MultipleRangeStaticProducer(request, StringIO(content), [(b'1', 1, 3), (b'2', 5, 1)])
    producer.start()
    self.assertEqual(b'1bcd2f', b''.join(request.written))