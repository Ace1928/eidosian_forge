import os
from twisted.internet import defer
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.web.http import NOT_FOUND
from twisted.web.script import PythonScript, ResourceScriptDirectory
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyRequest
from twisted.web.resource import Resource
class PythonScriptTests(TestCase):
    """
    Tests for L{PythonScript}.
    """

    def test_notFoundRender(self) -> defer.Deferred[None]:
        """
        If the source file a L{PythonScript} is initialized with doesn't exist,
        L{PythonScript.render} sets the HTTP response code to I{NOT FOUND}.
        """
        resource = PythonScript(self.mktemp(), None)
        request = DummyRequest([b''])
        d = _render(resource, request)

        def cbRendered(ignored: object) -> None:
            self.assertEqual(request.responseCode, NOT_FOUND)
        return d.addCallback(cbRendered)

    def test_renderException(self) -> defer.Deferred[None]:
        """
        L{ResourceScriptDirectory.getChild} returns a resource which renders a
        response with the HTTP 200 status code and the content of the rpy's
        C{request} global.
        """
        tmp = FilePath(self.mktemp())
        tmp.makedirs()
        child = tmp.child('test.epy')
        child.setContent(b'raise Exception("nooo")')
        resource = PythonScript(child._asBytesPath(), None)
        request = DummyRequest([b''])
        d = _render(resource, request)

        def cbRendered(ignored: object) -> None:
            self.assertIn(b'nooo', b''.join(request.written))
        return d.addCallback(cbRendered)