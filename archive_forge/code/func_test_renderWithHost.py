from twisted.internet.defer import gatherResults
from twisted.trial.unittest import TestCase
from twisted.web.http import NOT_FOUND
from twisted.web.resource import NoResource
from twisted.web.server import Site
from twisted.web.static import Data
from twisted.web.test._util import _render
from twisted.web.test.test_web import DummyRequest
from twisted.web.vhost import NameVirtualHost, VHostMonsterResource, _HostResource
def test_renderWithHost(self):
    """
        L{NameVirtualHost.render} returns the result of rendering the resource
        which is the value in the instance's C{host} dictionary corresponding
        to the key indicated by the value of the I{Host} header in the request.
        """
    virtualHostResource = NameVirtualHost()
    virtualHostResource.addHost(b'example.org', Data(b'winner', ''))
    request = DummyRequest([b''])
    request.requestHeaders.addRawHeader(b'host', b'example.org')
    d = _render(virtualHostResource, request)

    def cbRendered(ignored, request):
        self.assertEqual(b''.join(request.written), b'winner')
    d.addCallback(cbRendered, request)
    requestWithPort = DummyRequest([b''])
    requestWithPort.requestHeaders.addRawHeader(b'host', b'example.org:8000')
    dWithPort = _render(virtualHostResource, requestWithPort)

    def cbRendered(ignored, requestWithPort):
        self.assertEqual(b''.join(requestWithPort.written), b'winner')
    dWithPort.addCallback(cbRendered, requestWithPort)
    return gatherResults([d, dWithPort])