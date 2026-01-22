import os
from twisted.internet import defer
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.web.http import NOT_FOUND
from twisted.web.script import PythonScript, ResourceScriptDirectory
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyRequest
from twisted.web.resource import Resource
class ResourceScriptDirectoryTests(TestCase):
    """
    Tests for L{ResourceScriptDirectory}.
    """

    def test_renderNotFound(self) -> defer.Deferred[None]:
        """
        L{ResourceScriptDirectory.render} sets the HTTP response code to I{NOT
        FOUND}.
        """
        resource = ResourceScriptDirectory(self.mktemp())
        request = DummyRequest([b''])
        d = _render(resource, request)

        def cbRendered(ignored: object) -> None:
            self.assertEqual(request.responseCode, NOT_FOUND)
        return d.addCallback(cbRendered)

    def test_notFoundChild(self) -> defer.Deferred[None]:
        """
        L{ResourceScriptDirectory.getChild} returns a resource which renders an
        response with the HTTP I{NOT FOUND} status code if the indicated child
        does not exist as an entry in the directory used to initialized the
        L{ResourceScriptDirectory}.
        """
        path = self.mktemp()
        os.makedirs(path)
        resource = ResourceScriptDirectory(path)
        request = DummyRequest([b'foo'])
        child = resource.getChild('foo', request)
        d = _render(child, request)

        def cbRendered(ignored: object) -> None:
            self.assertEqual(request.responseCode, NOT_FOUND)
        return d.addCallback(cbRendered)

    def test_render(self) -> defer.Deferred[None]:
        """
        L{ResourceScriptDirectory.getChild} returns a resource which renders a
        response with the HTTP 200 status code and the content of the rpy's
        C{request} global.
        """
        tmp = FilePath(self.mktemp())
        tmp.makedirs()
        tmp.child('test.rpy').setContent(b"\nfrom twisted.web.resource import Resource\nclass TestResource(Resource):\n    isLeaf = True\n    def render_GET(self, request):\n        return b'ok'\nresource = TestResource()")
        resource = ResourceScriptDirectory(tmp._asBytesPath())
        request = DummyRequest([b''])
        child = resource.getChild(b'test.rpy', request)
        d = _render(child, request)

        def cbRendered(ignored: object) -> None:
            self.assertEqual(b''.join(request.written), b'ok')
        return d.addCallback(cbRendered)