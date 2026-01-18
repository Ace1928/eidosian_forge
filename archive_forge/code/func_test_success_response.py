from unittest import mock
import fixtures
from oslotest import base as test_base
import webob.dec
import webob.exc
from oslo_middleware import catch_errors
def test_success_response(self):

    @webob.dec.wsgify
    def application(req):
        return 'Hello, World!!!'
    self._test_has_request_id(application, webob.exc.HTTPOk.code)