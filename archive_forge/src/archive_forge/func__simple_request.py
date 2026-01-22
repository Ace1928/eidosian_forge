from unittest import mock
from oslo_config import cfg
from oslo_log import log
from oslo_messaging._drivers import common as rpc_common
import webob.exc
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.tests import utils
def _simple_request(self, path, params=None, method='GET'):
    environ = self._environ(path)
    environ['REQUEST_METHOD'] = method
    if params:
        qs = '&'.join(['='.join([k, str(params[k])]) for k in params])
        environ['QUERY_STRING'] = qs
    req = wsgi.Request(environ)
    req.context = utils.dummy_context('api_test_user', self.tenant)
    self.context = req.context
    return req