from unittest import mock
from openstack.block_storage.v3 import service
from openstack.tests.unit import base
def test_thaw(self):
    sot = service.Service(**EXAMPLE)
    res = sot.thaw(self.sess)
    self.assertIsNotNone(res)
    url = 'os-services/thaw'
    body = {'host': 'devstack'}
    self.sess.put.assert_called_with(url, json=body, microversion=self.sess.default_microversion)