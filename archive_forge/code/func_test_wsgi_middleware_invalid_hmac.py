from unittest import mock
from webob import response as webob_response
from osprofiler import _utils as utils
from osprofiler import profiler
from osprofiler.tests import test
from osprofiler import web
@mock.patch('osprofiler.web.profiler.init')
def test_wsgi_middleware_invalid_hmac(self, mock_profiler_init):
    hmac_key = 'secret'
    pack = utils.signed_pack({'base_id': '1', 'parent_id': '2'}, hmac_key)
    headers = {'a': '1', 'b': '2', 'X-Trace-Info': pack[0], 'X-Trace-HMAC': 'not valid hmac'}
    self._test_wsgi_middleware_with_invalid_trace(headers, hmac_key, mock_profiler_init)