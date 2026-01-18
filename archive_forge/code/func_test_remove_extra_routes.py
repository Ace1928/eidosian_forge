from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import router
from openstack.tests.unit import base
def test_remove_extra_routes(self):
    r = router.Router(**EXAMPLE)
    response = mock.Mock()
    response.headers = {}
    json_body = {'router': {}}
    response.body = json_body
    response.json = mock.Mock(return_value=response.body)
    response.status_code = 200
    sess = mock.Mock()
    sess.put = mock.Mock(return_value=response)
    ret = r.remove_extra_routes(sess, json_body)
    self.assertIsInstance(ret, router.Router)
    self.assertIsInstance(ret.routes, list)
    url = 'routers/IDENTIFIER/remove_extraroutes'
    sess.put.assert_called_with(url, json=json_body)