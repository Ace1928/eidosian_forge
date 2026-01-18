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
def test_renderDirectories(self):
    """
        L{static.DirectoryLister} is able to list all the directories inside
        a directory.
        """
    path = FilePath(self.mktemp())
    path.makedirs()
    path.child('dir1').makedirs()
    path.child('dir2 & 3').makedirs()
    lister = static.DirectoryLister(path.path)
    data = lister.render(self._request(b'foo'))
    body = b'<tr class="odd">\n    <td><a href="dir1/">dir1/</a></td>\n    <td></td>\n    <td>[Directory]</td>\n    <td></td>\n</tr>\n<tr class="even">\n    <td><a href="dir2%20%26%203/">dir2 &amp; 3/</a></td>\n    <td></td>\n    <td>[Directory]</td>\n    <td></td>\n</tr>'
    self.assertIn(body, data)