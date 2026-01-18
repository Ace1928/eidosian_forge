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
def test_renderFiles(self):
    """
        L{static.DirectoryLister} is able to list all the files inside a
        directory.
        """
    path = FilePath(self.mktemp())
    path.makedirs()
    path.child('file1').setContent(b'content1')
    path.child('file2').setContent(b'content2' * 1000)
    lister = static.DirectoryLister(path.path)
    data = lister.render(self._request(b'foo'))
    body = b'<tr class="odd">\n    <td><a href="file1">file1</a></td>\n    <td>8B</td>\n    <td>[text/html]</td>\n    <td></td>\n</tr>\n<tr class="even">\n    <td><a href="file2">file2</a></td>\n    <td>7K</td>\n    <td>[text/html]</td>\n    <td></td>\n</tr>'
    self.assertIn(body, data)