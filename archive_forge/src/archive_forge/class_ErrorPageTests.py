from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
class ErrorPageTests(TestCase):
    """
    Tests for L{_UnafeErrorPage}, L{_UnsafeNoResource}, and
    L{_UnsafeForbiddenResource}.
    """
    errorPage = ErrorPage
    noResource = NoResource
    forbiddenResource = ForbiddenResource

    def test_deprecatedErrorPage(self) -> None:
        """
        The public C{twisted.web.resource.ErrorPage} alias for the
        corresponding C{_Unsafe} class produces a deprecation warning when
        imported.
        """
        from twisted.web.resource import ErrorPage
        self.assertIs(ErrorPage, self.errorPage)
        [warning] = self.flushWarnings()
        self.assertEqual(warning['category'], DeprecationWarning)
        self.assertIn('twisted.web.pages.errorPage', warning['message'])

    def test_deprecatedNoResource(self) -> None:
        """
        The public C{twisted.web.resource.NoResource} alias for the
        corresponding C{_Unsafe} class produces a deprecation warning when
        imported.
        """
        from twisted.web.resource import NoResource
        self.assertIs(NoResource, self.noResource)
        [warning] = self.flushWarnings()
        self.assertEqual(warning['category'], DeprecationWarning)
        self.assertIn('twisted.web.pages.notFound', warning['message'])

    def test_deprecatedForbiddenResource(self) -> None:
        """
        The public C{twisted.web.resource.ForbiddenResource} alias for the
        corresponding C{_Unsafe} class produce a deprecation warning when
        imported.
        """
        from twisted.web.resource import ForbiddenResource
        self.assertIs(ForbiddenResource, self.forbiddenResource)
        [warning] = self.flushWarnings()
        self.assertEqual(warning['category'], DeprecationWarning)
        self.assertIn('twisted.web.pages.forbidden', warning['message'])

    def test_getChild(self) -> None:
        """
        The C{getChild} method of L{ErrorPage} returns the L{ErrorPage} it is
        called on.
        """
        page = self.errorPage(321, 'foo', 'bar')
        self.assertIdentical(page.getChild(b'name', object()), page)

    def _pageRenderingTest(self, page: Resource, code: int, brief: str, detail: str) -> None:
        request = DummyRequest([b''])
        template = '\n<html>\n  <head><title>%s - %s</title></head>\n  <body>\n    <h1>%s</h1>\n    <p>%s</p>\n  </body>\n</html>\n'
        expected = template % (code, brief, brief, detail)
        self.assertEqual(page.render(request), expected.encode('utf-8'))
        self.assertEqual(request.responseCode, code)
        self.assertEqual(request.responseHeaders, Headers({b'content-type': [b'text/html; charset=utf-8']}))

    def test_errorPageRendering(self) -> None:
        """
        L{ErrorPage.render} returns a C{bytes} describing the error defined by
        the response code and message passed to L{ErrorPage.__init__}.  It also
        uses that response code to set the response code on the L{Request}
        passed in.
        """
        code = 321
        brief = 'brief description text'
        detail = 'much longer text might go here'
        page = self.errorPage(code, brief, detail)
        self._pageRenderingTest(page, code, brief, detail)

    def test_noResourceRendering(self) -> None:
        """
        L{NoResource} sets the HTTP I{NOT FOUND} code.
        """
        detail = 'long message'
        page = self.noResource(detail)
        self._pageRenderingTest(page, NOT_FOUND, 'No Such Resource', detail)

    def test_forbiddenResourceRendering(self) -> None:
        """
        L{ForbiddenResource} sets the HTTP I{FORBIDDEN} code.
        """
        detail = 'longer message'
        page = self.forbiddenResource(detail)
        self._pageRenderingTest(page, FORBIDDEN, 'Forbidden Resource', detail)