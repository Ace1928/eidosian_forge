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
def test_singleRangeSets206PartialContent(self):
    """
        makeProducer when the Range header requests a single, satisfiable byte
        range sets the response code on the request to 'Partial Content'.
        """
    request = DummyRequest([])
    request.requestHeaders.addRawHeader(b'range', b'bytes=1-3')
    resource = self.makeResourceWithContent(b'abcdef')
    with resource.openForReading() as file:
        resource.makeProducer(request, file)
        self.assertEqual(http.PARTIAL_CONTENT, request.responseCode)