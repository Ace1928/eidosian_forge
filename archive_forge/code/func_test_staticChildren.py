from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_staticChildren(self) -> None:
    """
        L{Resource.putChild} adds a I{static} child to the resource.  That child
        is returned from any call to L{Resource.getChildWithDefault} for the
        child's path.
        """
    resource = Resource()
    child = Resource()
    sibling = Resource()
    resource.putChild(b'foo', child)
    resource.putChild(b'bar', sibling)
    self.assertIdentical(child, resource.getChildWithDefault(b'foo', DummyRequest([])))