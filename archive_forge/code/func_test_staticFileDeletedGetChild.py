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
def test_staticFileDeletedGetChild(self):
    """
        A L{static.File} created for a directory which does not exist should
        return childNotFound from L{static.File.getChild}.
        """
    staticFile = static.File(self.mktemp())
    request = DummyRequest([b'foo.bar'])
    child = staticFile.getChild(b'foo.bar', request)
    self.assertEqual(child, staticFile.childNotFound)