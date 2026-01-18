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
def test_undecodablePath(self):
    """
        A request whose path cannot be decoded as UTF-8 receives a not
        found response, and the failure is logged.
        """
    path = self.mktemp()
    if isinstance(path, bytes):
        path = path.decode('ascii')
    base = FilePath(path)
    base.makedirs()
    file = static.File(base.path)
    request = DummyRequest([b'\xff'])
    child = resource.getChildForRequest(file, request)
    d = self._render(child, request)

    def cbRendered(ignored):
        self.assertEqual(request.responseCode, 404)
        self.assertEqual(len(self.flushLoggedErrors(UnicodeDecodeError)), 1)
    d.addCallback(cbRendered)
    return d