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
def test_renderFiltered(self):
    """
        L{static.DirectoryLister} takes an optional C{dirs} argument that
        filter out the list of directories and files printed.
        """
    path = FilePath(self.mktemp())
    path.makedirs()
    path.child('dir1').makedirs()
    path.child('dir2').makedirs()
    path.child('dir3').makedirs()
    lister = static.DirectoryLister(path.path, dirs=['dir1', 'dir3'])
    data = lister.render(self._request(b'foo'))
    body = b'<tr class="odd">\n    <td><a href="dir1/">dir1/</a></td>\n    <td></td>\n    <td>[Directory]</td>\n    <td></td>\n</tr>\n<tr class="even">\n    <td><a href="dir3/">dir3/</a></td>\n    <td></td>\n    <td>[Directory]</td>\n    <td></td>\n</tr>'
    self.assertIn(body, data)