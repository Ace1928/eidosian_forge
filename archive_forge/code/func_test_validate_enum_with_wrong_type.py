import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
def test_validate_enum_with_wrong_type(self):

    class Version(object):
        number = types.Enum(str, 'v1', 'v2', None)

    class MyWS(WSRoot):

        @expose(str)
        @validate(Version)
        def setcplx(self, version):
            pass
    r = MyWS(['restjson'])
    app = webtest.TestApp(r.wsgiapp())
    res = app.post_json('/setcplx', params={'version': {'number': 1}}, expect_errors=True, headers={'Accept': 'application/json'})
    self.assertTrue(res.json_body['faultstring'].startswith("Invalid input for field/attribute number. Value: '1'. Value should be one of:"))
    self.assertIn('v1', res.json_body['faultstring'])
    self.assertIn('v2', res.json_body['faultstring'])
    self.assertIn('None', res.json_body['faultstring'])
    self.assertEqual(res.status_int, 400)