import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
def test_wsattr_default(self):

    class ComplexType(object):
        attr = wsme.types.wsattr(wsme.types.Enum(str, 'or', 'and'), default='and')

    class MyRoot(WSRoot):

        @expose(int)
        @validate(ComplexType)
        def clx(self, a):
            return a.attr
    r = MyRoot(['restjson'])
    app = webtest.TestApp(r.wsgiapp())
    res = app.post_json('/clx', params={}, expect_errors=True, headers={'Accept': 'application/json'})
    self.assertEqual(res.status_int, 400)