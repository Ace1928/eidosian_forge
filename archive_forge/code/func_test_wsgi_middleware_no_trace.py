from unittest import mock
from webob import response as webob_response
from osprofiler import _utils as utils
from osprofiler import profiler
from osprofiler.tests import test
from osprofiler import web
@mock.patch('osprofiler.web.profiler.init')
def test_wsgi_middleware_no_trace(self, mock_profiler_init):
    headers = {'a': '1', 'b': '2'}
    self._test_wsgi_middleware_with_invalid_trace(headers, 'secret', mock_profiler_init)