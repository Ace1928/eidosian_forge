from unittest import mock
from openstack.compute.v2 import service
from openstack import exceptions
from openstack.tests.unit import base
@mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=True)
def test_set_forced_down_after_211(self, mv_mock):
    sot = service.Service(**EXAMPLE)
    res = sot.set_forced_down(self.sess, 'host1', 'nova-compute', True)
    self.assertIsNotNone(res)
    url = 'os-services/force-down'
    body = {'binary': 'nova-compute', 'host': 'host1', 'forced_down': True}
    self.sess.put.assert_called_with(url, json=body, microversion='2.11')