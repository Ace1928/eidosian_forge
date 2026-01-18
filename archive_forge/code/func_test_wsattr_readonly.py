import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
def test_wsattr_readonly(self):

    class ComplexType(object):
        attr = wsme.types.wsattr(int, readonly=True)

    class MyRoot(WSRoot):

        @expose(int, body=ComplexType)
        @validate(ComplexType)
        def clx(self, a):
            return a.attr
    r = MyRoot(['restjson'])
    app = webtest.TestApp(r.wsgiapp())
    res = app.post_json('/clx', params={'attr': 1005}, expect_errors=True, headers={'Accept': 'application/json'})
    self.assertIn('Cannot set read only field.', res.json_body['faultstring'])
    self.assertIn('1005', res.json_body['faultstring'])
    self.assertEqual(res.status_int, 400)