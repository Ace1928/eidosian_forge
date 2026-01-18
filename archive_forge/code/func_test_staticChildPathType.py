from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_staticChildPathType(self) -> None:
    """
        Test that passing the wrong type to putChild results in a warning,
        and a failure in Python 3
        """
    resource = Resource()
    child = Resource()
    sibling = Resource()
    self.assertRaises(TypeError, resource.putChild, 'foo', child)
    self.assertRaises(TypeError, resource.putChild, None, sibling)