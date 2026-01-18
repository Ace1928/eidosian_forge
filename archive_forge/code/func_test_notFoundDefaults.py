from typing import cast
from twisted.trial.unittest import SynchronousTestCase
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.pages import errorPage, forbidden, notFound
from twisted.web.resource import IResource
from twisted.web.test.requesthelper import DummyRequest
def test_notFoundDefaults(self) -> None:
    """
        The default arguments to L{twisted.web.pages.notFound} produce
        a reasonable error page.
        """
    self.assertResponse(_render(notFound()), 404, b'<!DOCTYPE html>\n<html><head><title>404 - No Such Resource</title></head><body><h1>No Such Resource</h1><p>Sorry. No luck finding that resource.</p></body></html>')