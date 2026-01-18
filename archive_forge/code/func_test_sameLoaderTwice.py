import sys
from io import StringIO
from typing import List, Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, succeed
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.trial.util import suppress as SUPPRESS
from twisted.web._element import UnexposedMethodError
from twisted.web.error import FlattenerError, MissingRenderMethod, MissingTemplateLoader
from twisted.web.iweb import IRequest, ITemplateLoader
from twisted.web.server import NOT_DONE_YET
from twisted.web.template import (
from twisted.web.test._util import FlattenTestCase
from twisted.web.test.test_web import DummyRequest
def test_sameLoaderTwice(self) -> None:
    """
        Rendering the output of a loader, or even the same element, should
        return different output each time.
        """
    sharedLoader = XMLString('<p xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1"><t:transparent t:render="classCounter" /> <t:transparent t:render="instanceCounter" /></p>')

    class DestructiveElement(Element):
        count = 0
        instanceCount = 0
        loader = sharedLoader

        @renderer
        def classCounter(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
            DestructiveElement.count += 1
            return tag(str(DestructiveElement.count))

        @renderer
        def instanceCounter(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
            self.instanceCount += 1
            return tag(str(self.instanceCount))
    e1 = DestructiveElement()
    e2 = DestructiveElement()
    self.assertFlattensImmediately(e1, b'<p>1 1</p>')
    self.assertFlattensImmediately(e1, b'<p>2 2</p>')
    self.assertFlattensImmediately(e2, b'<p>3 1</p>')