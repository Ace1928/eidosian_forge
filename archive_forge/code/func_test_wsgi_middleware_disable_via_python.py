from unittest import mock
from webob import response as webob_response
from osprofiler import _utils as utils
from osprofiler import profiler
from osprofiler.tests import test
from osprofiler import web
@mock.patch('osprofiler.web.profiler.init')
def test_wsgi_middleware_disable_via_python(self, mock_profiler_init):
    request = mock.MagicMock()
    request.get_response.return_value = 'yeah!'
    web.disable()
    middleware = web.WsgiMiddleware('app', 'hmac_key', enabled=True)
    self.assertEqual('yeah!', middleware(request))
    self.assertEqual(mock_profiler_init.call_count, 0)