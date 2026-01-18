import datetime
from io import BytesIO, StringIO
from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver, MemoryReactor
from twisted.logger import (
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
from twisted.python.reflect import namedModule
from twisted.trial import unittest
from twisted.web import client, http, server, static, xmlrpc
from twisted.web.test.test_web import DummyRequest
from twisted.web.xmlrpc import (
def test_parseResponseWithoutData(self):
    """
        Some server can send a response without any data:
        L{QueryFactory.parseResponse} should catch the error and call the
        result errback.
        """
    content = '\n<methodResponse>\n <params>\n  <param>\n  </param>\n </params>\n</methodResponse>'
    d = self.queryFactory.deferred
    self.queryFactory.parseResponse(content)
    return self.assertFailure(d, IndexError)