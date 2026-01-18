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
@skipIf(sys.getfilesystemencoding().lower() not in ('utf-8', 'mcbs'), 'Cannot write unicode filenames with file system encoding of {}'.format(sys.getfilesystemencoding()))
def test_staticFileUnicodeFileName(self):
    """
        A request for a existing unicode file path encoded as UTF-8
        returns the contents of that file.
        """
    name = 'ῆ'
    content = b'content'
    base = FilePath(self.mktemp())
    base.makedirs()
    base.child(name).setContent(content)
    file = static.File(base.path)
    request = DummyRequest([name.encode('utf-8')])
    child = resource.getChildForRequest(file, request)
    d = self._render(child, request)

    def cbRendered(ignored):
        self.assertEqual(b''.join(request.written), content)
        self.assertEqual(request.responseHeaders.getRawHeaders(b'content-length')[0], networkString(str(len(content))))
    d.addCallback(cbRendered)
    return d