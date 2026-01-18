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
def test_parseResponseErrbackSafety(self):
    """
        We can safely call L{QueryFactory.clientConnectionLost} as an errback
        of L{QueryFactory.parseResponse}.
        """
    d = self.queryFactory.deferred
    d.addErrback(self.queryFactory.clientConnectionLost, self.reason)
    self.queryFactory.parseResponse(self.badContents)
    return d