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
def _makeFilePathWithStringIO(self):
    """
        Create a L{File} that when opened for reading, returns a L{StringIO}.

        @return: 2-tuple of the opened "file" and the L{File}.
        @rtype: L{tuple}
        """
    fakeFile = StringIO()
    path = FilePath(self.mktemp())
    path.touch()
    file = static.File(path.path)
    file.open = lambda: fakeFile
    return (fakeFile, file)